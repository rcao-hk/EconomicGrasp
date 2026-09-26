"""CPU-only ABBA benchmark of two CDF adapter source files on real frames.

No trainer/model is imported. Both adapter files and the source checkout are
read-only inputs. Timing excludes hashes, RNG capture, and reporting; all sample
and collated values plus post-sample RNG must match exactly across A/B/B/A.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from unittest.mock import patch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--baseline-adapter", required=True)
    parser.add_argument("--candidate-adapter", required=True)
    parser.add_argument("--output-json", "--outputjson", dest="output_json", required=True)
    parser.add_argument("--frames-per-split", type=int, default=2)
    parser.add_argument("--repo-root", help="Defaults to the source repository recorded in contract.model.file.")
    args = parser.parse_args()
    if args.frames_per_split < 1:
        parser.error("--frames-per-split must be positive")
    return args


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:4], "little")


def load_adapter(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load adapter source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.CVAExtendedLabelAdapter


def capture_cpu_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone()}


def restore_cpu_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    # Same CPU generator state as torch.manual_seed, without CUDA callbacks.
    torch.random.default_generator.manual_seed(seed)


def hash_tree(value):
    """Hash every nested value, including dtype, shape, sequence order and keys."""
    aggregate = hashlib.sha256()
    leaf_hashes = {}
    payload_bytes = 0

    def token(value):
        encoded = value.encode("utf-8")
        aggregate.update(len(encoded).to_bytes(8, "little"))
        aggregate.update(encoded)

    def visit(item, path):
        nonlocal payload_bytes
        if isinstance(item, dict):
            token("dict")
            token(str(len(item)))
            for key in sorted(item, key=str):
                token(type(key).__name__ + ":" + repr(key))
                visit(item[key], path + "/" + str(key))
            return
        if isinstance(item, (tuple, list)):
            token(type(item).__name__)
            token(str(len(item)))
            for i, member in enumerate(item):
                visit(member, path + "/" + str(i))
            return
        leaf = hashlib.sha256()
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                raise RuntimeError("GPU tensor encountered in CPU-only loader benchmark")
            tensor = item.detach().contiguous()
            metadata = f"torch:{tensor.dtype}:{tuple(tensor.shape)}"
            array = tensor.reshape(-1).view(torch.uint8).numpy()
            buffer = memoryview(array).cast("B")
            leaf.update(metadata.encode())
            leaf.update(buffer)
            payload_bytes += len(buffer)
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise TypeError("Object arrays are forbidden in the label benchmark")
            array = np.ascontiguousarray(item)
            metadata = f"numpy:{array.dtype.str}:{tuple(array.shape)}"
            buffer = memoryview(array).cast("B")
            leaf.update(metadata.encode())
            leaf.update(buffer)
            payload_bytes += len(buffer)
        elif isinstance(item, np.generic):
            leaf.update(("numpy_scalar:" + item.dtype.str).encode())
            leaf.update(item.tobytes())
        elif isinstance(item, (str, int, float, bool, bytes)) or item is None:
            leaf.update((type(item).__name__ + ":" + repr(item)).encode())
        else:
            raise TypeError(f"Unsupported value {type(item).__name__} at {path}")
        leaf_hashes[path] = leaf.hexdigest()
        token(leaf_hashes[path])

    visit(value, "root")
    return {"sha256": aggregate.hexdigest(), "payload_bytes": payload_bytes,
            "leaf_count": len(leaf_hashes), "leaf_sha256": leaf_hashes}


@contextlib.contextmanager
def count_npz_reads():
    counts = Counter()
    npz_class = np.lib.npyio.NpzFile
    original = npz_class.__getitem__

    def counted(instance, key):
        counts[str(key)] += 1
        return original(instance, key)

    with patch.object(npz_class, "__getitem__", counted):
        yield counts


def select_frames(contract, dataset, split, count):
    aliases = {"train"} if split == "train" else {"validation_test_seen", "validation", "heldout_test_diagnostic", "test_seen"}
    listed = [row for row in contract.get("probe_manifest", []) if row.get("split") in aliases]
    indices = list(dict.fromkeys(int(row["index"]) for row in listed))
    if len(indices) >= count:
        selected = [indices[int(i)] for i in np.linspace(0, len(indices) - 1, count, dtype=int)]
        source = "saved_contract_fixed_probes"
    else:
        scenes = {}
        for i, scene in enumerate(dataset.scenename):
            scenes.setdefault(scene, []).append(i)
        names = sorted(scenes)
        if count > len(names):
            raise ValueError(f"frames-per-split={count} exceeds the {split} scene count")
        selected = [scenes[names[int(i)]][len(scenes[names[int(i)]]) // 2]
                    for i in np.linspace(0, len(names) - 1, count, dtype=int)]
        source = "evenly_spaced_scenes_central_frame"
    return [{"split": split, "index": index, "scene": str(dataset.scenename[index]),
             "frame": int(dataset.frameid[index]), "selection": source} for index in selected]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    global np, torch
    args = parse_args()
    # Set this before importing Torch or project code. Never initialize CUDA.
    incoming_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    output = Path(args.output_json).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark result: {output}")
    contract_path = Path(args.contract).resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    cfg = contract["resolved_config"]
    source_metadata = contract["checkpoint"]["metadata"]
    fuse_depth = bool(source_metadata["use_fuse_depth"])
    if bool(cfg["use_fuse_depth"]) != fuse_depth:
        raise ValueError("Contract resolved use_fuse_depth conflicts with checkpoint metadata")
    if args.repo_root:
        repo = Path(args.repo_root).resolve()
    elif contract.get("model", {}).get("file"):
        repo = Path(contract["model"]["file"]).resolve().parent.parent
    else:
        repo = Path(__file__).resolve().parents[1]
    if not (repo / "dataset" / "graspnet_dataset.py").is_file():
        raise FileNotFoundError(f"Original dataset source not found under {repo}; provide --repo-root")
    sys.path.insert(0, str(repo))
    from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn

    paths = {"baseline": Path(args.baseline_adapter).resolve(), "candidate": Path(args.candidate_adapter).resolve()}
    classes = {name: load_adapter(path, "cdf_npz_benchmark_" + name) for name, path in paths.items()}
    label_folder = str(contract["data"]["label_cache"])
    base_kwargs = dict(root=cfg["dataset_root"], camera=cfg["camera"], voxel_size=cfg["voxel_size"],
                       num_points=cfg["num_point"], remove_outlier=True, augment=False, use_gt_depth=False,
                       use_fuse_depth=fuse_depth, graspness_mode=cfg["graspness_mode"],
                       min_depth=cfg["min_depth"], max_depth=cfg["max_depth"], bin_num=cfg["bin_num"],
                       depth_strides=1, extend_angle=True, load_grasp_payload=False)
    adapter_kwargs = dict(dataset_root=cfg["dataset_root"], use_cdf=True, label_folder=label_folder,
                          num_angle=cfg["num_angle"], num_depth=cfg["num_depth"])
    report = {"passed": False, "protocol": "per-frame ABBA: baseline,candidate,candidate,baseline",
              "timing_scope": "separate dataset __getitem__ and original collate_fn; hashes/RNG capture excluded",
              "equality_scope": "ALL nested sample and collated values, dtype/shape/order, and post-operation CPU RNG",
              "cuda_visible_devices": "", "incoming_cuda_visible_devices": incoming_cuda_visible_devices,
              "environment": {"python": sys.executable, "numpy": np.__version__, "torch": torch.__version__,
                              "torch_cpu_threads": torch.get_num_threads()},
              "contract": {"path": str(contract_path), "sha256": file_sha256(contract_path)},
              "sources": {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in paths.items()},
              "dataset_source": {"path": str(repo / "dataset" / "graspnet_dataset.py"),
                                  "sha256": file_sha256(repo / "dataset" / "graspnet_dataset.py")},
              "base_dataset_kwargs": base_kwargs, "adapter_kwargs": adapter_kwargs,
              "frames_per_split": args.frames_per_split, "initialization": [], "frames": [],
              "measurements": [], "mismatches": []}
    outer_rng = capture_cpu_rng()
    datasets = {}
    try:
        for split in ("train", "test_seen"):
            for variant, cls in classes.items():
                base = GraspNetMultiDataset(split=split, **base_kwargs)
                with count_npz_reads() as reads:
                    started = time.perf_counter()
                    adapter = cls(base, **adapter_kwargs)
                    seconds = time.perf_counter() - started
                datasets[split, variant] = adapter
                report["initialization"].append({"split": split, "variant": variant,
                    "adapter_seconds": seconds, "npz_getitem_per_key": dict(reads), "npz_getitem_total": sum(reads.values())})
            report["frames"] += select_frames(contract, datasets[split, "baseline"].base_dataset,
                                               split, args.frames_per_split)
        experiment_seed = int(contract.get("arguments", {}).get("seed", cfg.get("seed", 0)))
        for frame in report["frames"]:
            sample_seed = stable_seed(experiment_seed, "sample", 0, frame["index"])
            reference = None
            for order, variant in enumerate(("baseline", "candidate", "candidate", "baseline")):
                seed_cpu(sample_seed)
                pre_rng_hash = hash_tree(capture_cpu_rng())
                with count_npz_reads() as reads:
                    started = time.perf_counter()
                    sample = datasets[frame["split"], variant][frame["index"]]
                    sample_seconds = time.perf_counter() - started
                post_sample_rng_hash = hash_tree(capture_cpu_rng())
                started = time.perf_counter()
                collated = collate_fn([sample])
                collate_seconds = time.perf_counter() - started
                post_collate_rng_hash = hash_tree(capture_cpu_rng())
                # These potentially large hashes intentionally occur after both
                # timed sections. No returned key, including unused point inputs,
                # is dropped from the equivalence comparison.
                sample_hash = hash_tree(sample)
                collated_hash = hash_tree(collated)
                row = {**frame, "order": order, "variant": variant, "sample_seed": sample_seed,
                       "getitem_seconds": sample_seconds, "collate_seconds": collate_seconds,
                       "total_seconds": sample_seconds + collate_seconds,
                       "npz_getitem_per_key": dict(reads), "npz_getitem_total": sum(reads.values()),
                       "sample": sample_hash, "collated": collated_hash,
                       "pre_rng_sha256": pre_rng_hash["sha256"],
                       "post_sample_rng_sha256": post_sample_rng_hash["sha256"],
                       "post_collate_rng_sha256": post_collate_rng_hash["sha256"]}
                report["measurements"].append(row)
                if reference is None:
                    reference = row
                else:
                    for field in ("sample", "collated"):
                        if reference[field]["sha256"] != row[field]["sha256"]:
                            keys = reference[field]["leaf_sha256"].keys() | row[field]["leaf_sha256"].keys()
                            report["mismatches"].append({**frame, "order": order, "variant": variant, "field": field,
                                "different_leaves": [key for key in keys if reference[field]["leaf_sha256"].get(key)
                                                     != row[field]["leaf_sha256"].get(key)]})
                    for field in ("pre_rng_sha256", "post_sample_rng_sha256", "post_collate_rng_sha256"):
                        if reference[field] != row[field]:
                            report["mismatches"].append({**frame, "order": order, "variant": variant, "field": field})
                print(f"[ABBA] {frame['split']} {frame['scene']}/{frame['frame']:04d} {order} {variant}: "
                      f"getitem={sample_seconds:.3f}s collate={collate_seconds:.3f}s NPZ_reads={sum(reads.values())}", flush=True)
                del sample, collated
            write_report(output, report)
        by_variant = {}
        for variant in classes:
            rows = [row for row in report["measurements"] if row["variant"] == variant]
            by_variant[variant] = {"measurements": len(rows),
                "getitem_mean_s": statistics.mean(row["getitem_seconds"] for row in rows),
                "getitem_median_s": statistics.median(row["getitem_seconds"] for row in rows),
                "total_mean_s": statistics.mean(row["total_seconds"] for row in rows),
                "npz_getitem_counts": sorted({row["npz_getitem_total"] for row in rows})}
        report["summary"] = {"by_variant": by_variant,
            "mean_getitem_speedup": by_variant["baseline"]["getitem_mean_s"] / by_variant["candidate"]["getitem_mean_s"],
            "cache_caveat": "ABBA balances order; OS file cache and concurrent machine load are not controlled"}
        report["cuda_initialized"] = torch.cuda.is_initialized()
        if report["cuda_initialized"]:
            raise RuntimeError("Unexpected CUDA initialization in CPU-only benchmark")
        report["passed"] = not report["mismatches"]
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        restore_cpu_rng(outer_rng)
        write_report(output, report)
    if not report["passed"]:
        raise RuntimeError(f"Actual-data equivalence failed; inspect {output}")
    print(json.dumps({"passed": report["passed"], **report["summary"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
