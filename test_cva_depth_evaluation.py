"""CPU orchestration/metric tests; actual CUDA inference/AP needs the training host."""

import ast
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
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


GPU_FIXTURE = r'''
import json, os, sys, time
from pathlib import Path
import numpy as np
dump, event_dir, label, first, second, third, mode, split = sys.argv[1:]
dump, event_dir = Path(dump), Path(event_dir)
event_dir.mkdir(parents=True, exist_ok=True)
event = {"pid": os.getpid(), "gpu": os.environ["CUDA_VISIBLE_DEVICES"], "start": time.monotonic()}
(event_dir / (label + ".start.json")).write_text(json.dumps(event))
def wait_started(other):
    deadline = time.monotonic() + 15
    while not (event_dir / (other + ".start.json")).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("Fixture scheduler barrier timed out: " + other)
        time.sleep(.02)
if label == first:
    wait_started(second)
    if mode == "fail":
        print("fixture GPU inference failed", flush=True)
        sys.exit(7)
    wait_started(third)  # Requires a free GPU to pick up work before the first job finishes.
elif label == second:
    wait_started(first)
    if mode == "fail":
        while True:
            time.sleep(.1)  # The scheduler must terminate this sibling after the first job fails.
start_scene = {"test_seen": 100, "test_similar": 130, "test_novel": 160}[split]
for scene in range(start_scene, start_scene + 30):
    path = dump / ("scene_%04d" % scene) / "realsense" / "0000.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.empty((0, 17), dtype=np.float32))
event["end"] = time.monotonic()
(event_dir / (label + ".end.json")).write_text(json.dumps(event))
'''


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

    def gpu_fixture_factory(self, mode="success", splits=common.SPLITS):
        labels = [f"{variant}_{split}" for variant in common.VARIANTS for split in splits]
        def command(args, record, split, dump_dir):
            label = f"{Path(dump_dir).parent.name}_{split}"
            return [sys.executable, "-u", "-c", GPU_FIXTURE, str(dump_dir), str(self.root / "events"), label,
                    *labels[:3], mode, split]
        return command

    def test_gpu_queue_covers_all_variants_splits_once_and_reuses_finished_dumps(self):
        argv = self.infer_argv + ["--gpu_ids", "2,5", "--splits", ",".join(common.SPLITS)]
        with patch.object(inference, "inference_command", side_effect=self.gpu_fixture_factory()), redirect_stdout(io.StringIO()):
            inference.main(argv)
        events = [common.read_json(path) for path in (self.root / "events").glob("*.end.json")]
        self.assertEqual(len(events), 12)
        self.assertEqual({event["gpu"] for event in events}, {"2", "5"})
        # Real CPU subprocess lifetimes must overlap across devices, never on one device.
        self.assertTrue(any(a["start"] < b["start"] < a["end"] for a in events for b in events if a != b))
        for gpu_id in ("2", "5"):
            ordered = sorted((event for event in events if event["gpu"] == gpu_id), key=lambda event: event["start"])
            for previous, following in zip(ordered, ordered[1:]):
                self.assertLessEqual(previous["end"], following["start"])
        first = common.read_json(self.root / "events/base_test_seen.end.json")
        second = common.read_json(self.root / "events/base_test_similar.end.json")
        third = common.read_json(self.root / "events/base_test_novel.end.json")
        self.assertEqual(third["gpu"], second["gpu"])
        self.assertLess(third["start"], first["end"])
        for variant in common.VARIANTS:
            for split in common.SPLITS:
                manifest = common.read_completed_manifest(self.output / variant / split, variant, split)
                self.assertEqual(manifest["coverage"]["frames"], 30)
                self.assertIn(manifest["cuda_visible_devices"], ("2", "5"))
                self.assertNotIn("gpu_ids", manifest["identity"]["protocol"])
        with patch.object(inference.subprocess, "Popen") as process, redirect_stdout(io.StringIO()):
            inference.main(argv + ["--gpu_ids", "3,7"])
            process.assert_not_called()

    def test_gpu_failure_cancels_siblings_and_leaves_pending_jobs_unstarted(self):
        processes, actual_popen = [], subprocess.Popen
        def spawn(*args, **kwargs):
            process = actual_popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(inference, "inference_command", side_effect=self.gpu_fixture_factory("fail", ("test_seen",))), \
                patch.object(inference.subprocess, "Popen", side_effect=spawn), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "fixture GPU inference failed"):
                inference.main(self.infer_argv + ["--gpu_ids", "1,2"])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertFalse((self.output / "foreground").exists())
        for variant in ("base", "none"):
            manifest = common.read_json(self.output / variant / "test_seen/inference_manifest.json")
            self.assertEqual(manifest["status"], "running")

    def test_gpu_interrupt_reaps_children(self):
        processes, actual_popen = [], subprocess.Popen
        def spawn(*args, **kwargs):
            process = actual_popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(inference, "inference_command", side_effect=self.gpu_fixture_factory("fail", ("test_seen",))), \
                patch.object(inference.subprocess, "Popen", side_effect=spawn), \
                patch.object(inference.time, "sleep", side_effect=KeyboardInterrupt), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                inference.main(self.infer_argv + ["--gpu_ids", "1,2"])
        self.assertEqual(len(processes), 2)
        self.assertTrue(all(process.poll() is not None for process in processes))

    def test_gpu_selection_validation_and_no_silent_cpu_fallback(self):
        self.assertEqual(inference.parse_gpu_ids(" 1,2 "), ("1", "2"))
        for invalid in ("", "1,", "1,01", "-1", "1,1", "all"):
            with self.assertRaises(argparse.ArgumentTypeError):
                inference.parse_gpu_ids(invalid)
        args = inference.parse_args(self.infer_argv + ["--gpu_ids", "1,2"])
        record = inference.checkpoint_record(args.base_checkpoint, "base", args)
        self.assertIn("--require_cuda", inference.inference_command(args, record, "test_seen", self.output))
        tree = ast.parse((ROOT / "inference_cva_distill.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "inference")
        check = ast.Module(body=[function.body[0]], type_ignores=[])
        with self.assertRaisesRegex(RuntimeError, "Requested CUDA device is unavailable"):
            exec(compile(check, "real-cuda-guard", "exec"), {"DISTILL_INFER_ARGS": SimpleNamespace(require_cuda=True),
                 "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)), "os": os})


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

    def configured_combined_launcher(self, mode="all", fail_inference=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            for name in ("run_cva_depth_controls_eval.sh", "inference_cva_depth_controls.sh",
                         "eval_cva_depth_controls.sh", "depth_controls_eval_common.sh"):
                shutil.copyfile(ROOT / "scripts" / name, scripts / name)
            path = scripts / "run_cva_depth_controls_eval.sh"
            source = path.read_text(encoding="utf-8")
            overrides = {"RUN_MODE": mode, "DRY_RUN": "0" if fail_inference else "1", "GPU_IDS": "2,5",
                         "DATASET_ROOT": "/configured data with spaces", "OUTPUT_ROOT": "/configured output with spaces",
                         "BASE_CHECKPOINT": "/configured base with spaces.tar"}
            for key, value in overrides.items():
                source, count = re.subn(r"^" + key + r"=.*$", lambda match: key + "=" + shlex.quote(value), source, flags=re.MULTILINE)
                self.assertEqual(count, 1, key)
            path.write_text(source, encoding="utf-8", newline="\n")
            if fail_inference:
                (scripts / "inference_cva_depth_controls.sh").write_text("exit 7\n", encoding="utf-8")
                (scripts / "eval_cva_depth_controls.sh").write_text("printf AP_MUST_NOT_RUN\n", encoding="utf-8")
            # Deliberately conflicting inherited values: editing the file must be authoritative.
            env = dict(os.environ, DATASET_ROOT="/wrong data", GPU_IDS="99", INFER_BATCH_SIZE="1",
                       BASE_CHECKPOINT="/wrong base.tar", RUN_MODE="inference", DRY_RUN="0",
                       DEPTH_TEST_BASH=self.bash.replace("\\", "/"))
            env["PATH"] = str(Path(self.bash).parent) + os.pathsep + env.get("PATH", "")
            shell = 'bash() { "$DEPTH_TEST_BASH" "$@"; }\nexport -f bash\nbash "$@"'
            return subprocess.run([self.bash, "-c", shell, "configured-eval-test", str(path)], cwd=root,
                                  env=env, capture_output=True, text=True, timeout=30)

    def test_combined_launcher_uses_settings_in_file_and_supports_stage_selection(self):
        for mode, expected in (("all", 2), ("inference", 1), ("ap", 1)):
            result = self.configured_combined_launcher(mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            commands = [shlex.split(line.removeprefix("Running:")) for line in result.stdout.splitlines() if line.startswith("Running:")]
            self.assertEqual(len(commands), expected)
            for command in commands:
                self.assertEqual(command[command.index("--dataset_root") + 1], "/configured data with spaces")
            if mode != "ap":
                infer = commands[0]
                self.assertEqual(infer[infer.index("--gpu_ids") + 1], "2,5")
                self.assertEqual(infer[infer.index("--batch_size") + 1], "3")
                self.assertEqual(infer[infer.index("--base_checkpoint") + 1], "/configured base with spaces.tar")
            if mode != "inference":
                self.assertIn("eval_cva_depth_controls.py", commands[-1])
            if mode == "all":
                for flag in ("--prediction_root", "--variants", "--splits"):
                    self.assertEqual(commands[0][commands[0].index(flag) + 1], commands[1][commands[1].index(flag) + 1])

    def test_combined_launcher_does_not_run_ap_after_failed_inference(self):
        result = self.configured_combined_launcher(fail_inference=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertNotIn("AP_MUST_NOT_RUN", result.stdout)

    def test_new_bash_files_have_valid_syntax_and_lf(self):
        for filename in ("depth_controls_eval_common.sh", "inference_cva_depth_controls.sh",
                         "eval_cva_depth_controls.sh", "run_cva_depth_controls_eval.sh"):
            path = ROOT / "scripts" / filename
            self.assertNotIn(b"\r\n", path.read_bytes())
            run = subprocess.run([self.bash, "-n", str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
