"""CPU contract tests: 50%+50% training, 10% test, strict checkpoint binding.

Does NOT claim to validate historical CUDA, official evaluator or GPU runtime.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from rare_grasp.core import make_protocol, verify_protocol, digest, sha256, frame_key
from rare_grasp.runtime import check_mix10, check_mix_protocol, bind_config, read_bound_config
from rare_grasp.pipeline import check_model_matches_manifest, fit

REPO = Path(__file__).resolve().parents[1]


def fake_checkpoint(fraction=0.5, epoch=20, enable_gvar=True):
    frames = 12800 if fraction == 0.5 else 2600
    indices_hash = 'a' * 64
    design = {
        'enable_point_rayrope': True, 'enable_gvar': enable_gvar,
        'rayrope': {'group_chunk': 32, 'checkpoint_chunks': True}
    }
    if enable_gvar:
        design['gvar'] = {'action_chunk': 2024, 'activation_checkpoint': True}
    return {
        'epoch': epoch, 'use_cdf': True, 'pose_depth_mode': 'global_film',
        'use_fuse_depth': True, 'grasp_design': design,
        'gntrans_mix_protocol': {
            'original_train_count': frames, 'gntrans_train_count': frames,
            'mixed_train_count': frames * 2,
            'train_fraction_each_domain': fraction,
            'paired_scene_frame_sampling': True,
            'original_train_index_sha256': indices_hash,
            'gntrans_train_index_sha256': indices_hash,
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_mix50_scene_counts_and_no_leak(self):
        p = make_protocol(0.5)
        verify_protocol(p)
        self.assertEqual(p['mix_training_fraction'], 0.5)
        self.assertEqual(p['test_frame_fraction'], 0.1)
        self.assertEqual(len(p['train']), 25600)
        self.assertEqual(len(p['validation']), 40)
        self.assertEqual(len(p['test']), 4680)
        for domain in ('original', 'gntrans'):
            train = [r for r in p['train'] if r['domain'] == domain]
            self.assertEqual(len(train), 12800)
            for scene in range(100):
                frames = [r['anno'] for r in train if r['scene'] == scene]
                self.assertEqual(frames, list(range(0, 256, 2)))
            for split in ('test_seen', 'test_similar', 'test_novel'):
                rows = [r for r in p['test'] if r['domain'] == domain and r['split'] == split]
                self.assertEqual(len(rows), 780)
        train_keys = {frame_key(r) for r in p['train']}
        val_keys = {frame_key(r) for r in p['validation']}
        self.assertFalse(train_keys & val_keys)
        self.assertFalse(val_keys & {frame_key(r) for r in p['test']})

    def test_original_mix10_is_preserved(self):
        p = make_protocol()
        verify_protocol(p)
        self.assertNotIn('mix_training_fraction', p)
        self.assertEqual(len(p['train']), 5200)
        self.assertEqual(p['training_count_by_domain']['gntrans'], 2600)
        self.assertEqual(len(p['validation']), 40)
        self.assertEqual(len(p['test']), 4680)

    def test_tampered_manifest_fails(self):
        p = make_protocol(0.5)
        p['train'][0]['anno'] = 1
        with self.assertRaises(ValueError):
            verify_protocol(p)
        p = make_protocol(0.5)
        p['mix_training_fraction'] = 0.1
        with self.assertRaises(ValueError):
            verify_protocol(p)

    def test_wrong_model_fraction_is_rejected(self):
        check_model_matches_manifest({'mix_training_fraction': 0.5}, make_protocol(0.5))
        check_model_matches_manifest({}, make_protocol())
        with self.assertRaises(ValueError):
            check_model_matches_manifest({'mix_training_fraction': 0.1}, make_protocol(0.5))
        with self.assertRaises(ValueError):
            check_model_matches_manifest({'mix_training_fraction': 0.5}, make_protocol())

    def test_cli_manifest_fraction(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'manifest.json'
            r = subprocess.run([sys.executable, '-m', 'rare_grasp.pipeline', 'manifest',
                                '--train-fraction', '0.5', '--output', str(dest)],
                               cwd=REPO, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            verify_protocol(json.loads(dest.read_text()))
            self.assertEqual(len(json.loads(dest.read_text())['train']), 25600)


class CheckpointTests(unittest.TestCase):
    def test_strict_mix50_epoch20_and_optional_16(self):
        ck = fake_checkpoint()
        check_mix_protocol(ck, 0.5, 20)
        with self.assertRaises(ValueError):
            check_mix_protocol(ck, 0.5, 16)
        ck['epoch'] = 16
        check_mix_protocol(ck, 0.5, 16)
        with self.assertRaises(ValueError):
            check_mix_protocol(ck, 0.1, 16)

    def test_mixed10_weights_rejected_for_50(self):
        ck = fake_checkpoint(0.1)
        check_mix10(ck)
        with self.assertRaises(ValueError):
            check_mix_protocol(ck, 0.5, 20)
        ck = fake_checkpoint(0.5)
        with self.assertRaises(ValueError):
            check_mix10(ck)

    def test_no_invented_provenance(self):
        ck = fake_checkpoint()
        del ck['gntrans_mix_protocol']['paired_scene_frame_sampling']
        with self.assertRaises(ValueError):
            check_mix_protocol(ck, 0.5, 20)
        ck = fake_checkpoint()
        ck['gntrans_mix_protocol']['gntrans_train_index_sha256'] = 'b' * 64
        # Dataset-internal indices need not be equal when semantic frame pairs
        # were verified by the producer.
        check_mix_protocol(ck, 0.5, 20)
        del ck['gntrans_mix_protocol']['gntrans_train_index_sha256']
        with self.assertRaises(ValueError):
            check_mix_protocol(ck, 0.5, 20)

    def test_bind_syncs_recorded_memory_chunks_without_changing_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ck = fake_checkpoint()
            ckpath = td / 'model.tar'
            torch.save(ck, ckpath)
            template = td / 'template.json'
            template.write_text(json.dumps({
                'backbone': 'point_gvar', 'mix_training_fraction': 0.5,
                'model_args': [
                    '--enable_point_rayrope', '--enable_gvar',
                    '--rayrope_group_chunk', '2048', '--rayrope_checkpoint_chunks', '0',
                    '--gvar_action_chunk', '2048', '--gvar_activation_checkpoint', '0',
                ],
            }))
            out = td / 'bound.json'
            before = sha256(ckpath)
            bind_config(str(template), str(ckpath), str(out))
            bound = read_bound_config(str(out))
            self.assertEqual(bound['expected_checkpoint_epoch'], 20)
            self.assertEqual(bound['checkpoint_sha256'], before)
            self.assertEqual(bound['mix_training_fraction'], 0.5)
            args = bound['model_args']
            self.assertEqual(args[args.index('--rayrope_group_chunk') + 1], '32')
            self.assertEqual(args[args.index('--rayrope_checkpoint_chunks') + 1], '1')
            self.assertEqual(args[args.index('--gvar_action_chunk') + 1], '2024')
            self.assertEqual(args[args.index('--gvar_activation_checkpoint') + 1], '1')
            self.assertEqual(sha256(ckpath), before)
            # E15 must be explicitly requested; no silent epoch fallback.
            ck['epoch'] = 16
            historical_path = td / 'historical.tar'
            torch.save(ck, historical_path)
            with self.assertRaises(ValueError):
                bind_config(str(template), str(historical_path), str(td / 'rejected.json'))
            bind_config(str(template), str(historical_path), str(td / 'e15.json'),
                        allow_historical_e15=True)
            historical = read_bound_config(str(td / 'e15.json'))
            self.assertEqual(historical['expected_checkpoint_epoch'], 16)
            self.assertEqual(historical['provenance_status'], 'historical_e15_seen_monitored')


class TrainingChecks(unittest.TestCase):
    def test_formal_fit_rejects_incomplete_mix50(self):
        contract = {
            'config': {'mix_training_fraction': 0.5},
            'config_digest': 'digest', 'protocol_sha256': make_protocol(0.5)['protocol_sha256'],
            'families': 32, 'seed': 0
        }
        row = lambda part, anno: {
            'row': {'partition': part, 'domain': 'original', 'scene': 0, 'anno': anno}}
        tr = {'contract': contract, 'entries': [row('train', 0)]}
        val = {'contract': contract, 'entries': [row('validation', 5)]}
        args = type('Args', (), {'train_index': 'dummy', 'val_index': 'dummy', 'smoke': False})()
        with patch('rare_grasp.pipeline.load_index', side_effect=[tr, val]):
            with self.assertRaisesRegex(ValueError, '25600'):
                fit(args)

    def test_shell_syntax_and_binding_paths(self):
        script = REPO / 'scripts' / 'rare_mix50' / 'run.sh'
        r = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = script.read_text()
        self.assertIn('--mix_train_fraction 0.5', text)
        self.assertIn('--mix_eval_fraction 0.1', text)
        self.assertIn('--ckpt_save_interval 20', text)
        self.assertIn('checkpoint.tar', text)
        self.assertNotIn('checkpoint_19.tar', text)
        self.assertIn('rare_mix50', text)


if __name__ == '__main__':
    unittest.main()
