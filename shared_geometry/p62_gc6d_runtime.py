"""Real RGB snapshot / frozen correction / exact GC6D physics adapters.

Heavy dependencies are imported only on the server execution path. Official
GC6D eval_utils supplies collision detection and DexNet force closure. No NMS,
Top-K truncation, width clipping or GT background filtering is applied to banks.
GC6D's official eval_scene passes no synthetic table; this adapter does likewise.
"""
from __future__ import annotations

import copy
import inspect
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .p62_gc6d_contract import image_id, sha256, seed_for, quality_counts, cad_paths


@contextmanager
def no_tf32():
    import torch
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def read_rgb_frame(root, row, camera, crop_box=None):
    """Reads RGB + calibration only. Never opens depth, masks or scene_gt.json.

    Matches repository ToTensor -> tensor Resize(448) -> ImageNet normalization.
    Pose FiLM follows gc6d_dataset.build_camera_orientation_metadata's existing
    c2w interpretation of the release's cam_R_w2c key, not inferred BOP semantics.
    """
    from PIL import Image
    import torch
    import torch.nn.functional as F
    folder = Path(root) / "scenes" / f"{int(row['scene']):06d}"
    number = image_id(int(row["anno"]), camera)
    rgb = folder / "rgb" / f"{number:06d}.png"
    calibration = folder / "scene_camera.json"
    info = json.loads(calibration.read_text())[str(number)]
    image = np.asarray(Image.open(rgb).convert("RGB"), dtype=np.float32) / 255.
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if crop_box is None else tuple(crop_box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("Fixed RGB crop out of bounds")
    K = np.asarray(info["cam_K"], np.float32).reshape(3, 3).copy()
    if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    K[0, 2] -= x0; K[1, 2] -= y0
    K[0] *= 448. / (x1-x0); K[1] *= 448. / (y1-y0)
    R = np.asarray(info["cam_R_w2c"], np.float32).reshape(3, 3)
    if not np.isfinite(R).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-3) or np.linalg.det(R) < .999:
        raise ValueError("Invalid camera rotation")
    pose = -R[:, 2].copy(); gravity = R.T @ np.array([0, 0, 1], np.float32)
    pose /= np.linalg.norm(pose); gravity /= np.linalg.norm(gravity)
    tensor = torch.from_numpy(image[y0:y1, x0:x1].copy().transpose(2, 0, 1))[None]
    tensor = F.interpolate(tensor, size=(448, 448), mode="bilinear", align_corners=False, antialias=True)
    tensor = (tensor - torch.tensor([.485, .456, .406])[None, :, None, None]) / torch.tensor([.229, .224, .225])[None, :, None, None]
    arrays = dict(img=tensor,
                  K=torch.from_numpy(K)[None], camera_pose_vec=torch.from_numpy(pose)[None],
                  camera_gravity_vec=torch.from_numpy(gravity)[None],
                  camera_tilt_deg=torch.tensor([np.degrees(np.arccos(np.clip(pose[2], -1, 1)))], dtype=torch.float32))
    audit = dict(rgb=dict(path=str(rgb.resolve()), sha256=sha256(rgb)),
                 calibration=dict(path=str(calibration.resolve()), sha256=sha256(calibration)),
                 image_id=number, crop_box=[x0, y0, x1, y1], original_hw=[h, w],
                 pose_convention="repository_GC6D_json_R_c2w", sensor_depth_used=False,
                 gt_mask_used=False, gt_pose_used_by_rgb_model=False)
    return arrays, audit


class GC6DSource:
    """Reuse trusted NativeRuntime / FieldRuntime snapshot and replay, change input only."""
    def __init__(self, config, root, camera, device, crop_box=None):
        from interactive_grasp.native import NativeRuntime
        from .runtime import FieldRuntime
        import torch
        owner = self
        self.root, self.camera, self.device = root, camera, device
        self.crop_box, self.last_audit = crop_box, None

        class RGBNative(NativeRuntime):
            def sample(self, row, seed):
                value = seed_for(row, seed)
                random.seed(value); np.random.seed(value); torch.manual_seed(value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(value)
                batch, owner.last_audit = read_rgb_frame(owner.root, row, owner.camera, owner.crop_box)
                # NativeRuntime model can request explicitly named pose keys.
                for setting, key in (("camera_pose_key", "camera_pose_vec"),
                                     ("camera_gravity_key", "camera_gravity_vec")):
                    name = getattr(self.cfg, setting, key)
                    if name != key:
                        batch[name] = batch.pop(key)
                return {k: v.to(self.device) for k, v in batch.items()}

            def label(self, *args):
                raise RuntimeError("RGB source must not access quality labels")

            def sensor_keep(self, *args):
                raise RuntimeError("No sensor-depth prefilter in P6-2A")

        self.native = RGBNative(config, root, root, device)
        if getattr(self.native.cfg, "use_fuse_depth", False):
            raise ValueError("Fused-depth input is forbidden")
        self.runtime = FieldRuntime(self.native)
        self.provenance = dict(self.runtime.provenance, camera=camera,
                               sensor_prefilter="none", preprocessing="RGB_only_fixed_crop_448",
                               provider_source_sha256=sha256(__file__))

    def snapshot(self, row, seed):
        arrays, parity = self.runtime.snapshot(row, seed)
        return arrays, dict(input=self.last_audit, replay=parity)


class FrozenCorrector:
    def __init__(self, weight, device):
        from .p4_depth_baseline import load_corrector
        self.model, self.protocol, self.epoch = load_corrector(weight, device, smoke=False)
        self.model.eval().requires_grad_(False)
        self.device = device

    def predict(self, snapshots, feature_dtype):
        import torch
        if not snapshots:
            raise ValueError("Empty corrector batch")
        # Reproduce the actual P4 source storage quantization, not arbitrary fp16 compute.
        f = np.stack([x["feature_map"].astype(feature_dtype).astype(np.float32) for x in snapshots])
        d = np.stack([np.asarray(x["pred_depth"], np.float32) for x in snapshots])
        x = {"feature_map": torch.from_numpy(f).to(self.device),
             "pred_depth": torch.from_numpy(d).to(self.device)}
        with no_tf32(), torch.inference_mode():
            result = self.model(x).float().cpu().numpy()
        if result.shape != d.shape or not np.isfinite(result).all():
            raise ValueError("Corrector returned malformed metric depth")
        return result


def make_physics(snapshot, corrected_depth, bank, tau, band):
    import torch
    from .geometry import support_points, VisibleSurfaceTeacher
    # Do not import P1's training/cache builder or any clean targets here.
    points, local = support_points(torch.from_numpy(bank))
    xyz = points.reshape(-1, 3).numpy()
    def tokens(depth):
        v = VisibleSurfaceTeacher(depth, snapshot["K"], tau, band).label(xyz)
        x = np.concatenate((v["udf"][:, None], v["projective"][:, None],
                            np.eye(4, dtype=np.float32)[v["state"]]), axis=-1)
        return x.astype(np.float32).reshape(*bank.shape[:2], 75, 6)
    return dict(predicted_physics=tokens(snapshot["pred_depth"]),
                corrected_physics=tokens(corrected_depth),
                local_xyz=local.numpy().astype(np.float32),
                bank_dims=bank[..., 1:4].copy())



def _same_serialized_state(a, b, seen=None):
    """Namespace migration must preserve every numerical/cache state value."""
    seen = set() if seen is None else seen
    pair = (id(a), id(b))
    if pair in seen:
        return
    seen.add(pair)
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"):
            raise ValueError("DexNet namespace conversion altered an array")
    elif isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("DexNet namespace conversion changed state keys")
        for key in a:
            _same_serialized_state(a[key], b[key], seen)
    elif isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("DexNet namespace conversion changed a sequence")
        for x, y in zip(a, b):
            _same_serialized_state(x, y, seen)
    elif hasattr(a, "__dict__"):
        if not hasattr(b, "__dict__"):
            raise ValueError("DexNet namespace conversion lost object state")
        _same_serialized_state(vars(a), vars(b), seen)
    elif a != b:
        if not (isinstance(a, (float, np.floating)) and np.isnan(a) and np.isnan(b)):
            raise ValueError("DexNet namespace conversion changed a scalar")


def native_dexnet_object(obj, target_class):
    """Migrate a TRUSTED cached GraspNet object into GC6D's class namespace.

    The two APIs vendor DexNet classes and perform strict isinstance checks.
    Deserializing the already trusted object with mapped class names preserves
    SDF grids, mesh, transforms and cached samples WITHOUT recomputing geometry.
    Every serialized state value is checked, then actual label parity is still
    mandatory. This is not an untrusted-pickle loader or a type-check bypass.
    """
    if isinstance(obj, target_class):
        return obj
    import io
    import pickle
    old = "graspnetAPI.utils.dexnet."
    new = "graspclutter6dAPI.utils.dexnet."
    if not type(obj).__module__.startswith(old):
        raise TypeError("Unsupported cached DexNet object namespace: " + type(obj).__module__)

    class NamespaceReader(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith(old):
                module = new + module[len(old):]
            return super().find_class(module, name)

    converted = NamespaceReader(io.BytesIO(pickle.dumps(obj, protocol=4))).load()
    if not isinstance(converted, target_class):
        raise TypeError("Cached object is not GC6D GraspableObject3D after migration")
    _same_serialized_state(obj, converted)
    return converted


class ExactGC6D:
    """Official geometry/DexNet kernels, every input action retained in original order.

    Reference verification evaluates singleton actions through official eval_grasp;
    NMS/top10/top50 then cannot drop a candidate. Furniture/sensor/table geometry
    is NOT silently added. Cache is keyed by model ID, instances remain separate.
    """
    def __init__(self, root, camera, chunk=64, verify_per_frame=3):
        import open3d as o3d
        from graspclutter6dAPI import GraspGroup
        from graspclutter6dAPI.utils import eval_utils as eu
        from graspclutter6dAPI.utils.config import get_config
        self.root, self.camera = Path(root), camera
        self.eu, self.gg, self.o3d = eu, GraspGroup, o3d
        self.config = get_config()
        self.fc_list = np.array([1.2, 1., .8, .6, .4, .2])
        self.fc_configs = {}
        for mu in self.fc_list:
            c = copy.deepcopy(self.config["metrics"]["force_closure"])
            c["friction_coef"] = round(float(mu), 2)
            self.fc_configs[round(float(mu), 2)] = eu.GraspQualityConfigFactory.create_config(c)
        if chunk < 1 or verify_per_frame < 1:
            raise ValueError("Positive chunk and reference parity budget required")
        self.chunk, self.verify_per_frame = int(chunk), int(verify_per_frame)
        self.cache = {}
        self._converted = {}
        self.assets = {}
        self.provenance = dict(api="graspclutter6dAPI", eval_utils_sha256=sha256(eu.__file__),
            quality_config=self.config, force_closure="official_get_grasp_score",
            table_policy="none_official_GC6D", background_filter=False, nms=False,
            topk=False, width_clipping=False, voxel_size_m=.008, empty_thresh=10,
            collision_outlier_m=.05, adapter_sha256=sha256(__file__),
            reference="singleton_official_eval_grasp_no_candidate_truncation",
            dexnet_namespace_policy="trusted_state_preserving_GN_to_GC_checked")
        # Pin dependencies used by official functions (not only the wrapper module).
        for obj in (get_config, eu.PointGraspMetrics3D, eu.ParallelJawPtGrasp3D,
                    eu.GraspQualityConfigFactory, eu.matrix_to_dexnet_params,
                    eu.GraspableObject3D, eu.ObjFile, eu.SdfFile):
            path = inspect.getfile(obj)
            self.provenance.setdefault("physics_dependency_hashes", {})[str(Path(path).resolve())] = sha256(path)

    def native_object(self, obj):
        if isinstance(obj, self.eu.GraspableObject3D):
            return obj
        if id(obj) not in self._converted:
            self._converted[id(obj)] = (obj, native_dexnet_object(obj, self.eu.GraspableObject3D))
        return self._converted[id(obj)][1]

    def model(self, obj_id):
        if obj_id not in self.cache:
            p = self.root / "models_m" / f"obj_{obj_id:06d}.ply"
            raw = np.asarray(self.o3d.io.read_point_cloud(str(p)).points)
            if raw.ndim != 2 or raw.shape[1] != 3 or len(raw) == 0 or not np.isfinite(raw).all():
                raise ValueError(f"Invalid metre-scale CAD point cloud {p}")
            self.assets[str(p.resolve())] = sha256(p)
            # Official cache preference, but read-only. Only use trusted dataset pickles.
            paths = cad_paths(self.root, obj_id)
            for q in paths[1:]:
                self.assets[str(q.resolve())] = sha256(q)
            if paths[1].suffix == ".pkl":
                import pickle
                with open(paths[1], "rb") as f:
                    dex = pickle.load(f)
            else:
                dex = self.eu.load_dexnet_model(str(paths[1].with_suffix("")))
            dex = self.native_object(dex)
            self.cache[obj_id] = (self.eu.voxel_sample_points(raw, .008), dex)
        return self.cache[obj_id]

    def scene(self, row):
        file = self.root / "scenes" / f"{int(row['scene']):06d}" / "scene_gt.json"
        gt = json.loads(file.read_text())[str(image_id(int(row["anno"]), self.camera))]
        models, dex, poses = [], [], []
        objects = []
        for obj in gt:
            obj_id = int(obj["obj_id"])
            points, graspable = self.model(obj_id)
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(obj["cam_R_m2c"]).reshape(3, 3)
            pose[:3, 3] = np.asarray(obj["cam_t_m2c"]).reshape(3) / 1000.
            if not np.isfinite(pose).all() or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3):
                raise ValueError("Bad GC6D metre-scale model-to-camera pose")
            models.append(points); dex.append(graspable); poses.append(pose); objects.append(obj_id)
        if not models:
            raise ValueError("Frame has no annotated objects")
        paths = {str(file.resolve()): sha256(file)}
        for obj_id in set(objects):
            for item in cad_paths(self.root, obj_id):
                path = str(item.resolve())
                paths[path] = self.assets[path]
        return models, dex, poses, dict(object_ids=objects, files=paths)

    def evaluate_arrays(self, grasps, models, dex, poses, table=None):
        """Also used for GraspNet parity, where an explicit table is supplied."""
        g = np.asarray(grasps, np.float32)
        if g.ndim != 2 or g.shape[1] != 17 or not len(g) or not np.isfinite(g).all():
            raise ValueError("Expected nonempty finite exact N x 17 actions")
        if len(models) != len(dex) or len(models) != len(poses) or not models:
            raise ValueError("Mismatched model instances / poses")
        dex = [self.native_object(x) for x in dex]
        cam = [self.eu.transform_points(m, p) for m, p in zip(models, poses)]
        cloud = np.concatenate(cam)
        seg = np.concatenate([np.full(len(m), i, np.int64) for i, m in enumerate(cam)])
        assigned = np.concatenate([seg[self.eu.compute_closest_points(g[s:s+self.chunk, 13:16], cloud)]
                                   for s in range(0, len(g), self.chunk)])
        scene = cloud if table is None else np.concatenate([cloud, table])
        collision = np.zeros(len(g), bool); empty = collision.copy()
        friction = np.full(len(g), -1., np.float32)
        for obj in range(len(models)):
            indices = np.flatnonzero(assigned == obj)
            for s in range(0, len(indices), self.chunk):
                ids = indices[s:s+self.chunk]
                col, emp, dg = self.eu.collision_detection([g[ids]], [cam[obj]], [dex[obj]],
                    [poses[obj]], scene, outlier=.05, empty_thresh=10, return_dexgrasps=True)
                collision[ids] = np.asarray(col[0], bool); empty[ids] = np.asarray(emp[0], bool)
                for j, idx in enumerate(ids):
                    if not collision[idx] and dg[0][j] is not None:
                        friction[idx] = self.eu.get_grasp_score(dg[0][j], dex[obj], self.fc_list, self.fc_configs)
        return dict(assigned_obj=assigned, collision_or_empty=collision, empty=empty,
                    friction=friction, quality_count=quality_counts(friction, collision))

    def verify(self, grasps, labels, models, dex, poses, table=None):
        dex = [self.native_object(x) for x in dex]
        ids = np.unique(np.linspace(0, len(grasps)-1, min(self.verify_per_frame, len(grasps))).astype(int))
        positive = 0
        for idx in ids:
            gs, fs, cs = self.eu.eval_grasp(self.gg(grasps[idx:idx+1].copy()), models, dex,
                                           poses, copy.deepcopy(self.config), table=table, voxel_size=.008, TOP_K=1)
            selected = [(g, f, c) for g, f, c in zip(gs, fs, cs) if len(g)]
            if len(selected) != 1 or len(selected[0][0]) != 1:
                raise RuntimeError("Official singleton evaluator dropped/duplicated the action")
            g, f, c = selected[0]
            if not np.array_equal(np.asarray(g)[0, 1:16], grasps[idx, 1:16]):
                raise RuntimeError("Reference evaluator altered physical action")
            q = quality_counts(np.asarray(f), np.asarray(c, bool))[0]
            if (q != labels["quality_count"][idx] or bool(c[0]) != labels["collision_or_empty"][idx]
                or not np.isclose(f[0], labels["friction"][idx], atol=1e-6, rtol=0)):
                raise RuntimeError(f"Exact-action reference mismatch at candidate {idx}")
            positive += int(q > 0)
        return dict(n_checked=len(ids), n_positive=positive, mismatches=0, indices=ids.tolist())

    def evaluate(self, row, grasps):
        models, dex, poses, source = self.scene(row)
        labels = self.evaluate_arrays(grasps, models, dex, poses)
        parity = self.verify(grasps, labels, models, dex, poses)
        return labels, dict(source=source, reference_parity=parity)
