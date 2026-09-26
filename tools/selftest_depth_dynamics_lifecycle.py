"""Event and output-history invariants, requiring only the Python stdlib."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_cva_depth_dynamics as trainer


def advance(observations, state=None):
    state = {} if state is None else state
    for step, fraction in observations:
        state = trainer.advance_flat_event_state(state, step=step, fraction_flat=fraction)
    return state


class EventLifecycleTest(unittest.TestCase):
    def test_initial_flatness_is_not_training_induced_event(self):
        state = advance([(0, 1.), (100, 1.), (200, .9), (300, 1.)])
        self.assertTrue(state["initial_flat"])
        self.assertFalse(state["nonflat_established"])
        for name in ("first_anomaly", "first_threshold", "confirmed_event"):
            self.assertIsNone(state[name])

    def test_initially_flat_then_sustained_nonflat_then_lost(self):
        state = advance([(0, 1.), (100, .7), (200, .4), (300, .2),
                         (400, .3), (500, .9), (600, 1.), (700, .9)])
        self.assertTrue(state["initial_flat"])
        self.assertTrue(state["nonflat_established"])
        self.assertEqual(state["first_nonflat_step"], 100)
        self.assertEqual(state["nonflat_confirmed_step"], 300)
        self.assertEqual(state["first_anomaly"], 400)
        self.assertEqual(state["first_threshold"], 500)
        self.assertEqual(state["confirmed_event"], 700)
        self.assertEqual(state["event_origin"], "lost_after_nonflat_emergence")

    def test_transient_nonflat_probe_does_not_establish_structure(self):
        state = advance([(0, 1.), (100, .2), (200, .3), (300, .9), (400, .9), (500, .9)])
        self.assertFalse(state["nonflat_established"])
        self.assertIsNone(state["confirmed_event"])

    def test_initial_nonflat_trajectory_preserves_event_thresholds(self):
        state = advance([(0, .1), (100, .1), (200, .2), (300, .8), (400, .9), (500, 1.)])
        self.assertFalse(state["initial_flat"])
        self.assertEqual(state["first_anomaly"], 200)  # Existing initial flat images are not a new anomaly.
        self.assertEqual(state["first_threshold"], 300)
        self.assertEqual(state["confirmed_event"], 500)
        self.assertEqual(state["event_origin"], "loss_of_initial_nonflat_structure")

    def test_missing_eligible_images_stay_unknown_and_break_streaks(self):
        state = advance([(0, None), (100, None)])
        self.assertIsNone(state["initial_flat"])
        state = advance([(200, 1.), (300, .1), (400, .1), (500, None), (600, .1)], state)
        self.assertEqual(state["initial_observation_step"], 200)
        self.assertFalse(state["nonflat_established"])
        self.assertEqual(state["consecutive_nonflat"], 1)

    def test_same_update_is_not_counted_twice(self):
        state = advance([(0, 0.), (100, 1.)])
        repeated = advance([(100, 1.)], state)
        self.assertEqual(state, repeated)
        self.assertEqual(repeated["consecutive_thresholds"], 1)
        with self.assertRaisesRegex(ValueError, "conflicting/reordered"):
            advance([(100, .5)], state)

    def test_state_roundtrip_retains_nonflat_and_collapse_streaks(self):
        prefix = [(0, 1.), (100, .1), (200, .1)]
        suffix = [(300, .1), (400, .9), (500, .9), (600, .9)]
        resumed = json.loads(json.dumps(advance(prefix)))
        self.assertEqual(advance(prefix + suffix), advance(suffix, resumed))

    def test_legacy_warm_state_keeps_existing_timestamps(self):
        legacy = {"initial_flat": False, "first_anomaly": 100, "first_threshold": 200,
                  "confirmed_event": None, "consecutive_thresholds": 2}
        state = advance([(400, 1.)], legacy)
        self.assertEqual(state["first_threshold"], 200)
        self.assertEqual(state["confirmed_event"], 400)
        self.assertTrue(state["nonflat_established"])
        self.assertIn("migration_note", state)

    def test_legacy_flat_state_does_not_invent_past_emergence(self):
        legacy = {"initial_flat": True, "first_anomaly": None, "first_threshold": None,
                  "confirmed_event": None, "consecutive_thresholds": 0}
        state = advance([(500, .1), (600, .1)], legacy)
        self.assertFalse(state["nonflat_established"])
        self.assertEqual(state["first_nonflat_step"], 500)

    def test_legacy_warm_unknown_reference_starts_with_observation_baseline(self):
        legacy = {"initial_flat": False, "first_anomaly": None, "first_threshold": None,
                  "confirmed_event": None, "consecutive_thresholds": 0}
        state = advance([(400, None), (500, 1 / 16)], legacy)
        self.assertTrue(state["migration_reference_unknown"])
        self.assertEqual(state["migration_observation_baseline_step"], 500)
        self.assertEqual(state["migration_observation_baseline_fraction"], 1 / 16)
        self.assertEqual(state["reference_flat_fraction"], 1 / 16)
        self.assertIsNone(state["first_nonflat_step"])
        self.assertIsNone(state["nonflat_confirmed_step"])
        self.assertIsNone(state["first_anomaly"])
        later = advance([(600, 2 / 16)], json.loads(json.dumps(state)))
        self.assertEqual(later["first_anomaly"], 600)
        self.assertEqual(later["migration_observation_baseline_step"], 500)


class ResumeOutputSafetyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "arm"
        self.diag = Path(self.directory.name) / "diagnostics"
        self.output.mkdir()
        self.diag.mkdir()

    def write(self, relative, text, output=False):
        path = (self.output if output else self.diag) / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def check(self, step=100):
        return trainer.assert_resume_outputs_not_newer(self.output, self.diag, step)

    def test_logs_at_or_before_checkpoint_are_allowed(self):
        self.write("train_steps.jsonl", '{"step":50}\n{"step":100}\n')
        self.write("events.jsonl", '{"step":100,"initial_flat":false}\n')
        self.write("route_connectivity.csv", "step,loss\n100,depth\n")
        self.write("local_directional_probe.csv", "batch_id,loss\n0,depth\n")
        self.write("latest_checkpoint.json", '{"step":100}', output=True)
        self.write("contract.json", '{"arguments":{"max_steps":2000},"completion":{"step":100}}')
        self.assertEqual(self.check()["status"], "no_newer_history")

    def test_newer_jsonl_blocks_resume_without_touching_any_files(self):
        path = self.write("train_steps.jsonl", '{"step":100}\n{"step":110}\n')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "newer history"):
            self.check()
        self.assertEqual(path.read_bytes(), before)

    def test_newer_csv_blocks_resume(self):
        self.write("native_autograd_directional.csv", "step,loss\n200,depth\n")
        with self.assertRaisesRegex(ValueError, "step 200"):
            self.check()

    def test_deleted_newer_checkpoint_history_still_blocks_resume(self):
        self.write("checkpoints_manifest.json", '[{"step":200,"deleted_after_retention":true}]')
        with self.assertRaisesRegex(ValueError, "step 200"):
            self.check()

    def test_newer_snapshot_without_manifest_blocks_resume(self):
        self.write("checkpoints/step_000200.partial", "unfinished", output=True)
        with self.assertRaisesRegex(ValueError, "step 200"):
            self.check()

    def test_malformed_tail_fails_closed(self):
        self.write("fixed_probe.jsonl", '{"step":100}\n{"step":')
        with self.assertRaisesRegex(ValueError, "malformed log"):
            self.check()

    def test_malformed_csv_or_column_counts_fail_closed(self):
        for contents in ('step,loss\n100,"unfinished', "step,loss\n100\n",
                         "step,loss\n100,depth,extra\n", "batch_id,loss\n0\n",
                         "step,step\n100,100\n"):
            with self.subTest(contents=contents):
                self.write("probe.csv", contents)
                with self.assertRaisesRegex(ValueError, "malformed log"):
                    self.check()

    def test_checkpoint_step_must_be_nonnegative_integer(self):
        for step in (100.5, -1, True, None, float("nan"), float("inf"), "100"):
            with self.subTest(step=step):
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    self.check(step)

    def test_guard_runs_before_trainer_or_manifest_writes(self):
        path = self.write("fixed_probe.jsonl", '{"step":200}\n')
        snapshot = {str(p): p.read_bytes() for folder in (self.output, self.diag)
                    for p in folder.rglob("*") if p.is_file()}
        loads = []

        def fake_load(checkpoint, **kwargs):
            loads.append(checkpoint)
            return {"format_version": trainer.FORMAT_VERSION, "step": 100}

        args = SimpleNamespace(output=str(self.output), diagnostics_dir=str(self.diag),
                               resume_checkpoint="older_checkpoint.pt")
        with patch.object(trainer, "torch", SimpleNamespace(load=fake_load), create=True):
            with self.assertRaisesRegex(ValueError, "newer history"):
                trainer.Experiment(args, [])
        self.assertEqual(loads, ["older_checkpoint.pt"])
        after = {str(p): p.read_bytes() for folder in (self.output, self.diag)
                 for p in folder.rglob("*") if p.is_file()}
        self.assertEqual(snapshot, after)
        self.assertTrue(path.exists())

    def test_rescue_cannot_silently_resume_as_ordinary_training(self):
        args = SimpleNamespace(output=str(self.output), diagnostics_dir=str(self.diag),
                               resume_checkpoint="rescue.pt", mode="train")
        before = {str(p): p.read_bytes() for folder in (self.output, self.diag)
                  for p in folder.rglob("*") if p.is_file()}
        for branch in ("normal", "remove_view_fixed_clip"):
            state = {"format_version": trainer.FORMAT_VERSION, "step": 100,
                     "arguments": {"rescue_branch": branch}}
            with patch.object(trainer, "torch", SimpleNamespace(load=lambda *a, **k: state), create=True):
                with self.assertRaisesRegex(ValueError, "explicit --branch"):
                    trainer.Experiment(args, [])
            trainer.assert_rescue_resume_explicit(state, SimpleNamespace(mode="audit"))
            trainer.assert_rescue_resume_explicit(state, SimpleNamespace(mode="train", rescue_branch="normal"))
        trainer.assert_rescue_resume_explicit({"arguments": {}}, args)
        after = {str(p): p.read_bytes() for folder in (self.output, self.diag)
                 for p in folder.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
