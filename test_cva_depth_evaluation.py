"""CPU orchestration/metric tests; actual CUDA inference/AP needs the training host."""

import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import cva_depth_evaluation as common
import eval_cva_depth_controls as evaluation
import inference_cva_depth_controls as inference


ROOT = Path(__file__).resolve().parent


def full_checkpoint(variant):
    return {"model_state_dict": {"fixture": torch.tensor(1.)}, "distill_contract_version": 2,
            "distill_stage": 1, "geometry_depth_source": "pred", "seed_selection_mode": "image_fps",
            "depth_head_executed": True, "legacy_dataset_use_gt_depth": False, "pose_depth_mode": "global_film",
            "use_fuse_depth": True, "epoch": 16 if variant == "base" else 5,
            "depth_geometry_contract_version": 1,
            "depth_geometry_args": {} if variant == "base" else {"variant": variant, "train_scope": "joint"}}


def write_fixture_dumps(command, log_path=None):
    def value(flag):
        return command[command.index(flag) + 1]
    stride = round(1 / float(value("--sample_interval")))
    for scene in common.scene_ids(value("--test_mode")):
        for anno in common.annotation_ids(stride):
            path = Path(value("--save_dir")) / f"scene_{scene:04d}" / value("--camera") / f"{anno:04d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.empty((0, 17), dtype=np.float32))


class FixtureAPI:
    calls = []

    def __init__(self, root, camera, split):
        self.split = split

    def eval_seen(self, dump_folder, proc=2, anno_sample_ratio=1.0):
        self.calls.append((dump_folder, proc, anno_sample_ratio))
        variant = Path(dump_folder).parent.name
        # Distinct control improvements; friction channels cannot be confused.
        offset = {"base": 0., "none": .01, "foreground": .03, "anchor": .07}[variant]
        shape = (30, len(common.annotation_ids(round(1 / anno_sample_ratio))), 50, 6)
        result = np.broadcast_to(np.arange(1, 7) * .1 + offset, shape).copy()
        return result, result.mean()


class DepthEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "predictions with spaces"
        self.controls = self.root / "controls"
        for variant in common.VARIANTS:
            path = self.root / "base checkpoint.tar" if variant == "base" else self.controls / variant / "checkpoint.tar"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(full_checkpoint(variant), path)
        self.infer_argv = ["--dataset_root", str(self.root / "data"), "--prediction_root", str(self.output),
                           "--base_checkpoint", str(self.root / "base checkpoint.tar"), "--controls_dir", str(self.controls),
                           "--frame_stride", "256", "--splits", "test_seen"]
        self.eval_argv = ["--dataset_root", str(self.root / "data"), "--prediction_root", str(self.output), "--splits", "test_seen"]

    def tearDown(self):
        self.temp.cleanup()

    def generate(self, argv=None):
        with redirect_stdout(io.StringIO()), patch.object(inference, "run_logged", side_effect=write_fixture_dumps) as child:
            inference.main(argv or self.infer_argv)
        return child.call_count

    def test_commands_match_real_inference_frame_selection_and_checkpoint_contract(self):
        args = inference.parse_args(self.infer_argv + ["--frame_stride", "10", "--topk_views", "4"])
        record = inference.checkpoint_record(args.base_checkpoint, "base", args)
        command = inference.inference_command(args, record, "test_seen", self.output)
        self.assertEqual(command[command.index("--sample_interval") + 1], "0.1")
        for flag in ("--use_fuse_depth", "--use_top4_view_infer", "--use_cdf", "--extend_angle"):
            self.assertIn(flag, command)
        self.assertNotIn("--use_obs_depth", command)
        self.assertEqual(command[command.index("--checkpoint_path") + 1], str(self.root / "base checkpoint.tar"))
        # Exercise the real legacy entry point's selector without importing CUDA extensions.
        tree = ast.parse((ROOT / "inference_cva_distill.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_subset")
        node.returns = None
        for argument in node.args.args:
            argument.annotation = None
        namespace = {"Subset": torch.utils.data.Subset}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "real-frame-selector", "exec"), namespace)
        for stride in (1, 7, 10, 256):
            _, indices = namespace["_build_subset"](list(range(512)), 1 / stride)
            expected = common.annotation_ids(stride) + [256 + n for n in common.annotation_ids(stride)]
            self.assertEqual(indices, expected)
        wrong = full_checkpoint("base")
        wrong["geometry_depth_source"] = "gt"
        torch.save(wrong, args.base_checkpoint)
        with self.assertRaisesRegex(ValueError, "RGB/pred-depth"):
            inference.checkpoint_record(args.base_checkpoint, "base", args)
        with self.assertRaisesRegex(ValueError, "does not belong"):
            inference.checkpoint_record(self.controls / "none" / "checkpoint.tar", "anchor", args)

    def test_complete_inference_skips_and_protocol_change_fails_before_running(self):
        self.assertEqual(self.generate(), 4)
        self.assertEqual(self.generate(), 0)
        with self.assertRaisesRegex(ValueError, "Different checkpoint/protocol"):
            self.generate(self.infer_argv + ["--topk_views", "4"])
        with redirect_stdout(io.StringIO()):
            evaluation.main(self.eval_argv + ["--check_only"])

    def test_interrupted_split_recomputes_and_incomplete_dump_cannot_be_evaluated(self):
        argv = self.infer_argv + ["--variants", "none"]
        with patch.object(inference, "run_logged", side_effect=RuntimeError("interrupted")), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                inference.main(argv)
        manifest = self.output / "none/test_seen/inference_manifest.json"
        self.assertEqual(common.read_json(manifest)["status"], "running")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluation.prepare_jobs(evaluation.parse_args(self.eval_argv + ["--variants", "none"]))
        self.assertEqual(self.generate(argv), 1)

    def test_dump_validation_rejects_missing_extra_corrupt_but_accepts_empty(self):
        args = inference.parse_args(self.infer_argv)
        record = inference.checkpoint_record(args.base_checkpoint, "base", args)
        command = inference.inference_command(args, record, "test_seen", self.output)
        write_fixture_dumps(command)
        check = lambda: common.check_dumps(self.output, "test_seen", "realsense", 256)
        self.assertEqual(check()["empty_grasp_frames"], 30)
        path = self.output / "scene_0100/realsense/0000.npy"
        np.save(path, np.zeros((1, 16)))
        with self.assertRaisesRegex(ValueError, "N x 17"):
            check()
        np.save(path, np.full((1, 17), np.nan))
        with self.assertRaisesRegex(ValueError, "N x 17"):
            check()
        np.save(path, np.empty((0, 17)))
        extra = path.with_name("0001.npy")
        np.save(extra, np.empty((0, 17)))
        with self.assertRaisesRegex(ValueError, "extra=1"):
            check()
        extra.unlink()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing=1"):
            check()

    def test_checkpoint_overwritten_by_training_is_not_marked_complete(self):
        def mutate_checkpoint(command, log_path):
            write_fixture_dumps(command)
            path = command[command.index("--checkpoint_path") + 1]
            checkpoint = full_checkpoint("none")
            checkpoint["model_state_dict"]["fixture"] += 1
            torch.save(checkpoint, path)
        with patch.object(inference, "run_logged", side_effect=mutate_checkpoint), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "Checkpoint changed during inference"):
                inference.main(self.infer_argv + ["--variants", "none"])
        manifest = common.read_json(self.output / "none/test_seen/inference_manifest.json")
        self.assertEqual(manifest["status"], "running")

    def test_official_api_full_evaluation_and_sampled_fork_dispatch(self):
        class Upstream:
            def eval_seen(self, dump_folder, proc=2):
                return dump_folder, proc
        self.assertEqual(evaluation.call_evaluator(Upstream(), "test_seen", "fixture", 3, 1), ("fixture", 3))
        with self.assertRaisesRegex(RuntimeError, "anno_sample_ratio"):
            evaluation.call_evaluator(Upstream(), "test_seen", "fixture", 3, 10)
        class Broken:
            def eval_seen(self, dump_folder, proc=2, anno_sample_ratio=1.):
                raise TypeError("error inside evaluator")
        with self.assertRaisesRegex(TypeError, "inside evaluator"):
            evaluation.call_evaluator(Broken(), "test_seen", "fixture", 3, 1)

    def test_ap_metrics_cache_and_control_differences(self):
        self.generate()
        FixtureAPI.calls = []
        with patch.dict(sys.modules, {"graspnetAPI": SimpleNamespace(GraspNetEval=FixtureAPI)}), redirect_stdout(io.StringIO()):
            evaluation.main(self.eval_argv)
            self.assertEqual(len(FixtureAPI.calls), 4)
            evaluation.main(self.eval_argv)
            self.assertEqual(len(FixtureAPI.calls), 4)
            evaluation.main(self.eval_argv + ["--force"])
            self.assertEqual(len(FixtureAPI.calls), 8)
        summary = common.read_json(self.output / "ap_summary.json")
        self.assertAlmostEqual(summary["results"][0]["ap_percent"], 35.)
        self.assertAlmostEqual(summary["results"][0]["ap_04_percent"], 20.)
        self.assertAlmostEqual(summary["results"][0]["ap_08_percent"], 40.)
        self.assertEqual(len(summary["results"][0]["scene_ap_percent"]), 30)
        differences = {row["comparison"]: row["ap_pp"] for row in summary["deltas_percentage_points"]}
        self.assertAlmostEqual(differences["none - base"], 1.)
        self.assertAlmostEqual(differences["foreground - none"], 2.)
        self.assertAlmostEqual(differences["anchor - foreground"], 4.)
        self.assertTrue((self.output / "ap_summary.csv").is_file())
        self.assertTrue((self.output / "ap_deltas.csv").is_file())

    def test_wrong_evaluator_coverage_and_ap_are_rejected(self):
        result = np.zeros((30, 1, 50, 6))
        with self.assertRaisesRegex(ValueError, "shape"):
            common.summarize_accuracy(result, "test_seen", 10, 0.)
        with self.assertRaisesRegex(ValueError, "differs"):
            common.summarize_accuracy(result, "test_seen", 256, .1)
        result[0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            common.summarize_accuracy(result, "test_seen", 256, 0.)

    def test_cli_help_is_available_without_graspnet(self):
        for filename in ("inference_cva_depth_controls.py", "eval_cva_depth_controls.py"):
            run = subprocess.run([sys.executable, str(ROOT / filename), "--help"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("--prediction_root", run.stdout)


class BashEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = os.environ.get("DEPTH_TEST_BASH") or shutil.which("bash")
        if not cls.bash:
            raise unittest.SkipTest("Bash unavailable.")

    def launch(self, script, **extra):
        env = {k: v for k, v in os.environ.items() if k not in ("BASE_CHECKPOINT", "CHECKPOINT", "CONTROLS_DIR")}
        env.update(DATASET_ROOT="/data with spaces", PREDICTION_ROOT="/pred with spaces", DRY_RUN="1",
                   OUTPUT_ROOT="/training with spaces", RUN_TAG="e5", CHECK_ONLY="0", FORCE_EVAL="0",
                   DEPTH_TEST_BASH=self.bash.replace("\\", "/"))
        env.update(extra)
        env["PATH"] = str(Path(self.bash).parent) + os.pathsep + env.get("PATH", "")
        command = 'bash() { "$DEPTH_TEST_BASH" "$@"; }\nexport -f bash\nbash "$@"'
        run = subprocess.run([self.bash, "-c", command, "depth-evaluation-test", f"scripts/{script}"],
                             cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        return [shlex.split(line.removeprefix("Running:")) for line in run.stdout.splitlines() if line.startswith("Running:")]

    def test_launchers_preserve_paths_and_evaluation_needs_no_checkpoint(self):
        command, = self.launch("inference_cva_depth_controls.sh", CHECKPOINT="/base with spaces.tar", FRAME_STRIDE="10", TOPK_VIEWS="4")
        self.assertEqual(command[command.index("--base_checkpoint") + 1], "/base with spaces.tar")
        self.assertEqual(command[command.index("--controls_dir") + 1], "/training with spaces/controls_e5")
        self.assertEqual(command[command.index("--frame_stride") + 1], "10")
        command, = self.launch("eval_cva_depth_controls.sh", CHECK_ONLY="1", EVAL_NUM_WORKERS="6")
        self.assertIn("--check_only", command)
        self.assertEqual(command[command.index("--num_workers") + 1], "6")
        self.assertNotIn("--base_checkpoint", command)
        infer, evaluate = self.launch("run_cva_depth_controls_eval.sh", BASE_CHECKPOINT="/override base.tar")
        self.assertIn("inference_cva_depth_controls.py", infer)
        self.assertIn("eval_cva_depth_controls.py", evaluate)
        self.assertEqual(infer[infer.index("--base_checkpoint") + 1], "/override base.tar")
        for flag in ("--prediction_root", "--variants", "--splits"):
            self.assertEqual(infer[infer.index(flag) + 1], evaluate[evaluate.index(flag) + 1])

    def test_new_bash_files_have_valid_syntax_and_lf(self):
        for filename in ("depth_controls_eval_common.sh", "inference_cva_depth_controls.sh",
                         "eval_cva_depth_controls.sh", "run_cva_depth_controls_eval.sh"):
            path = ROOT / "scripts" / filename
            self.assertNotIn(b"\r\n", path.read_bytes())
            run = subprocess.run([self.bash, "-n", str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
