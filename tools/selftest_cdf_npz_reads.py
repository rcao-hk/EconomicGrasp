"""Compressed-CDF loader regression: metadata checks must not decode payloads.

Run with the project Python. This uses real NPZ files and simulates the old
NumPy Mapping.__contains__ behavior even when testing on a newer NumPy.
"""
from collections import Counter
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.cdf_label_adapter import CVAExtendedLabelAdapter


class FrameInputs:
    load_grasp_payload = False
    extend_angle = True
    scenename = ["scene_0000"]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        assert index == 0
        return {"object_poses_list": [np.eye(3, 4, dtype=np.float32),
                                      np.eye(3, 4, dtype=np.float32) * 2],
                "frame_witness": np.array([17, 23], dtype=np.int64)}


class CompressedLabelReadsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "labels").mkdir()
        self.path = self.root / "labels" / "scene_0000_labels.npz"
        self.arrays = {
            "points": np.arange(12, dtype=np.float32).reshape(4, 3),
            "pointid": np.array([1, 0, 1, 0], dtype=np.int64),
            "vgraspness": np.arange(12, dtype=np.float32).reshape(4, 3) / 12,
            "topview": np.tile(np.array([0, 1], dtype=np.int32), (4, 1)),
            "extend_angle": np.array(1), "num_angle": np.array(2), "num_depth": np.array(2),
            "cdf_bins": (np.arange(32) % 4).astype(np.uint8).reshape(4, 2, 2, 2),
            "cdf_thresholds": np.array([.2, .4, .6], dtype=np.float32),
            "widths_depth_mm": np.arange(1, 33, dtype=np.uint16).reshape(4, 2, 2, 2),
            "width_valids_depth": (np.arange(32) % 2).astype(np.uint8).reshape(4, 2, 2, 2),
            "unused_payload": np.zeros((1024,), dtype=np.float32),
        }
        np.savez_compressed(self.path, **self.arrays)

    def adapter(self):
        return CVAExtendedLabelAdapter(FrameInputs(), str(self.root), use_cdf=True,
                                       label_folder="labels", num_angle=2, num_depth=2)

    @contextlib.contextmanager
    def count_reads(self):
        npz_class = np.lib.npyio.NpzFile
        original_getitem = npz_class.__getitem__
        reads = Counter()
        contains = Counter()

        def counted_getitem(instance, key):
            reads[key] += 1
            return original_getitem(instance, key)

        def legacy_contains(instance, key):
            contains[key] += 1
            try:
                instance[key]
            except KeyError:
                return False
            return True

        with patch.object(npz_class, "__getitem__", counted_getitem), \
             patch.object(npz_class, "__contains__", legacy_contains):
            yield reads, contains

    def test_archive_metadata_is_lazy_and_sample_reads_once(self):
        with self.count_reads() as (reads, contains):
            adapter = self.adapter()
            self.assertEqual(reads, Counter(extend_angle=1, num_angle=1, num_depth=1))
            self.assertEqual(contains, {})
            reads.clear()
            with contextlib.redirect_stdout(io.StringIO()):
                sample = adapter[0]
            required = adapter._COMMON_KEYS + adapter._CDF_KEYS
            self.assertEqual(reads, Counter({key: 1 for key in required}))
            self.assertEqual(contains, {})
        np.testing.assert_array_equal(sample["frame_witness"], [17, 23])
        for obj_id in range(2):
            mask = self.arrays["pointid"] == obj_id
            for output, source in (("grasp_points_list", "points"),
                                   ("view_graspness_list", "vgraspness"),
                                   ("top_view_index_list", "topview"),
                                   ("grasp_cdf_bins_list", "cdf_bins"),
                                   ("grasp_widths_depth_list", "widths_depth_mm"),
                                   ("grasp_width_valids_depth_list", "width_valids_depth")):
                np.testing.assert_array_equal(sample[output][obj_id], self.arrays[source][mask])
                self.assertEqual(sample[output][obj_id].dtype, self.arrays[source].dtype)
        np.testing.assert_array_equal(sample["cdf_thresholds"], self.arrays["cdf_thresholds"])

    def test_missing_key_fails_without_loading_any_arrays(self):
        self.arrays.pop("cdf_bins")
        np.savez_compressed(self.path, **self.arrays)
        with self.count_reads() as (reads, contains):
            with self.assertRaisesRegex(KeyError, "cdf_bins"):
                self.adapter()
            self.assertEqual(reads, {})
            self.assertEqual(contains, {})


if __name__ == "__main__":
    unittest.main()
