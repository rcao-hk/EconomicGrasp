# generate_tsdf_scene_depth.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import numpy as np
from PIL import Image
import scipy.io as scio
import open3d as o3d
import copy
import traceback
import multiprocessing as mp
import cv2


def depth_m_to_points(depth_m, K, max_points=100000):
    """
    depth_m: (H,W), meters
    K: (3,3)
    return: (N,3) camera-coordinate points
    """
    H, W = depth_m.shape
    valid = depth_m > 0

    ys, xs = np.where(valid)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - float(K[0, 2])) * z / float(K[0, 0])
    y = (ys.astype(np.float32) - float(K[1, 2])) * z / float(K[1, 1])

    pts = np.stack([x, y, z], axis=1).astype(np.float32)

    if max_points is not None and max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]

    return pts


def make_colored_pcd(points, color):
    """
    color: [r,g,b] in [0,1]
    """
    pcd = o3d.geometry.PointCloud()
    if points.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        colors = np.tile(np.asarray(color, dtype=np.float64)[None, :], (points.shape[0], 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def merge_pcds(pcds):
    merged = o3d.geometry.PointCloud()
    for pcd in pcds:
        merged += pcd
    return merged


def load_virtual_depth_m(cfg, scene, camera_type, ann_id, factor_depth):
    if cfg.virtual_dataset_root is None:
        virtual_root = os.path.join(cfg.dataset_root, "virtual_scenes")
    else:
        virtual_root = cfg.virtual_dataset_root

    path = os.path.join(
        virtual_root,
        scene,
        camera_type,
        f"{ann_id:04d}_depth.png",
    )

    if not os.path.exists(path):
        print(f"[Vis][warn] missing virtual depth: {path}")
        return None

    depth_raw = np.array(Image.open(path))
    return depth_raw.astype(np.float32) / float(factor_depth)


def save_depth_triplet_vis(
    cfg,
    scene,
    camera_type,
    ann_id,
    K,
    factor_depth,
    fused_depth_m,
    obs_depth_raw,
):
    """
    Save one PLY containing:
      red   = fused/rendered TSDF depth
      green = observed sensor depth
      blue  = virtual_scenes depth
    """
    obs_depth_m = obs_depth_raw.astype(np.float32) / float(factor_depth)
    virtual_depth_m = load_virtual_depth_m(cfg, scene, camera_type, ann_id, factor_depth)

    fused_pts = depth_m_to_points(
        fused_depth_m, K, max_points=cfg.vis_max_points_per_cloud
    )
    obs_pts = depth_m_to_points(
        obs_depth_m, K, max_points=cfg.vis_max_points_per_cloud
    )

    pcds = [
        make_colored_pcd(fused_pts, [1.0, 0.0, 0.0]),  # red
        make_colored_pcd(obs_pts, [0.0, 1.0, 0.0]),    # green
    ]

    if virtual_depth_m is not None:
        virtual_pts = depth_m_to_points(
            virtual_depth_m, K, max_points=cfg.vis_max_points_per_cloud
        )
        pcds.append(make_colored_pcd(virtual_pts, [0.0, 0.0, 1.0]))  # blue

    merged = merge_pcds(pcds)

    save_dir = os.path.join(cfg.vis_root, scene, camera_type)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{ann_id:04d}_fused_obs_virtual.ply")
    o3d.io.write_point_cloud(save_path, merged)
    print(
        f"[Vis] saved {save_path} | "
        f"fused={fused_pts.shape[0]}, obs={obs_pts.shape[0]}"
    )
    
    
def get_o3d_integration_module():
    # compatibility for different Open3D versions
    if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "integration"):
        return o3d.pipelines.integration
    return o3d.integration


def scene_name(scene_id: int) -> str:
    return f"scene_{scene_id:04d}"


def load_meta(meta_path):
    meta = scio.loadmat(meta_path)
    K = meta["intrinsic_matrix"].astype(np.float64)
    factor_depth = float(np.asarray(meta["factor_depth"]).squeeze())
    return K, factor_depth


def make_intrinsic_o3d(width, height, K):
    return o3d.camera.PinholeCameraIntrinsic(
        int(width),
        int(height),
        float(K[0, 0]),
        float(K[1, 1]),
        float(K[0, 2]),
        float(K[1, 2]),
    )


def get_paths(dataset_root, scene, camera_type, ann_id):
    base = os.path.join(dataset_root, "scenes", scene, camera_type)
    return {
        "rgb": os.path.join(base, "rgb", f"{ann_id:04d}.png"),
        "depth": os.path.join(base, "depth", f"{ann_id:04d}.png"),
        "meta": os.path.join(base, "meta", f"{ann_id:04d}.mat"),
    }


def load_pose(dataset_root, scene, camera_type, ann_id, use_align=True):
    base = os.path.join(dataset_root, "scenes", scene, camera_type)

    camera_poses = np.load(os.path.join(base, "camera_poses.npy"))
    camera_pose = camera_poses[ann_id].astype(np.float64)

    if use_align:
        align_mat = np.load(os.path.join(base, "cam0_wrt_table.npy")).astype(np.float64)
        # same convention as your graspness generation / workspace-mask script
        T_world_cam = align_mat @ camera_pose
    else:
        T_world_cam = camera_pose

    return T_world_cam


def make_rgbd(color_path, depth_path, factor_depth, depth_trunc):
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=float(factor_depth),
        depth_trunc=float(depth_trunc),
        convert_rgb_to_intensity=False,
    )
    return rgbd


def clean_mesh(mesh):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def integrate_scene_tsdf(cfg):
    integration = get_o3d_integration_module()
    scene = scene_name(cfg.scene_id)

    volume = integration.ScalableTSDFVolume(
        voxel_length=float(cfg.voxel_length),
        sdf_trunc=float(cfg.sdf_trunc),
        color_type=integration.TSDFVolumeColorType.NoColor,
        volume_unit_resolution=int(cfg.volume_unit_resolution),
        depth_sampling_stride=int(cfg.depth_sampling_stride),
    )

    frame_ids = list(range(cfg.start_ann, cfg.end_ann, cfg.integrate_stride))
    print(f"[TSDF] scene={scene}, camera={cfg.camera_type}, integrate frames={len(frame_ids)}")

    for k, ann_id in enumerate(frame_ids):
        paths = get_paths(cfg.dataset_root, scene, cfg.camera_type, ann_id)

        if not os.path.exists(paths["depth"]):
            print(f"[skip] missing depth: {paths['depth']}")
            continue

        K, factor_depth = load_meta(paths["meta"])
        depth_img = Image.open(paths["depth"])
        width, height = depth_img.size

        intrinsic = make_intrinsic_o3d(width, height, K)
        rgbd = make_rgbd(paths["rgb"], paths["depth"], factor_depth, cfg.depth_trunc)

        T_world_cam = load_pose(
            cfg.dataset_root,
            scene,
            cfg.camera_type,
            ann_id,
            use_align=not cfg.no_align,
        )
        T_cam_world = np.linalg.inv(T_world_cam)

        volume.integrate(rgbd, intrinsic, T_cam_world)

        if (k + 1) % cfg.log_every == 0 or k == 0:
            print(f"[TSDF] integrated {k + 1}/{len(frame_ids)} frames, ann={ann_id:04d}")

    mesh = volume.extract_triangle_mesh()
    mesh = clean_mesh(mesh)

    if cfg.mesh_smooth:
        print(f"[TSDF] apply Taubin mesh smoothing: iter={cfg.mesh_smooth_iter}")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=int(cfg.mesh_smooth_iter))
        mesh = clean_mesh(mesh)
    
    mesh_dir = os.path.join(cfg.output_root, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_path = os.path.join(mesh_dir, f"{scene}_{cfg.camera_type}_tsdf.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)

    print(f"[TSDF] mesh saved: {mesh_path}")
    print(f"[TSDF] vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")

    return mesh, mesh_path


def build_raycast_scene(mesh):
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(tmesh)
    return ray_scene


def render_depth_from_mesh(ray_scene, K, T_cam_world, width, height, depth_trunc):
    """
    Return z-depth in camera frame, meters.
    Important:
      RaycastingScene returns t_hit along ray direction.
      To get depth-map convention, we compute hit point in world coordinates
      and then transform it back to camera frame, taking z.
    """
    K_tensor = o3d.core.Tensor(K.astype(np.float32))
    ext_tensor = o3d.core.Tensor(T_cam_world.astype(np.float32))

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=K_tensor,
        extrinsic_matrix=ext_tensor,
        width_px=int(width),
        height_px=int(height),
    )

    ans = ray_scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    rays_np = rays.numpy()

    valid = np.isfinite(t_hit)
    points_w = rays_np[..., :3] + rays_np[..., 3:] * t_hit[..., None]

    ones = np.ones((*points_w.shape[:2], 1), dtype=np.float32)
    points_w_h = np.concatenate([points_w.astype(np.float32), ones], axis=-1)

    points_c = points_w_h @ T_cam_world.astype(np.float32).T
    depth_m = points_c[..., 2]

    depth_m[~valid] = 0.0
    depth_m[depth_m <= 0] = 0.0
    depth_m[depth_m > depth_trunc] = 0.0

    return depth_m.astype(np.float32)


def save_depth_png_from_meters(depth_m, save_path, depth_scale=1.0):
    """
    depth_m: rendered depth in meters, float32, shape (H, W)
    save format:
        depth_raw = depth_m * 1000.0 / depth_scale
        uint16 PNG

    If depth_scale=1.0:
        saved value is millimeter depth, i.e. depth_m * 1000.
    """
    depth_np = np.asarray(depth_m, dtype=np.float32)

    # invalid / negative -> 0
    depth_np[~np.isfinite(depth_np)] = 0.0
    depth_np[depth_np < 0] = 0.0

    depth_np = depth_np * 1000.0 / float(depth_scale)
    depth_np = np.rint(depth_np)
    depth_np = np.clip(depth_np, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ok = cv2.imwrite(save_path, depth_np)
    if not ok:
        raise IOError(f"Failed to write depth image: {save_path}")

    return depth_np


def render_scene_depths(cfg, mesh):
    scene = scene_name(cfg.scene_id)
    ray_scene = build_raycast_scene(mesh)

    save_dir = os.path.join(cfg.output_root, scene, cfg.camera_type)
    os.makedirs(save_dir, exist_ok=True)

    mae_list = []
    valid_ratio_list = []

    frame_ids = list(range(cfg.start_ann, cfg.end_ann, cfg.render_stride))
    print(f"[Render] scene={scene}, camera={cfg.camera_type}, render frames={len(frame_ids)}")
    print(f"[Render] save_dir={save_dir}")

    for k, ann_id in enumerate(frame_ids):
        paths = get_paths(cfg.dataset_root, scene, cfg.camera_type, ann_id)
        K, factor_depth = load_meta(paths["meta"])

        sensor_raw = np.array(Image.open(paths["depth"]))
        height, width = sensor_raw.shape

        T_world_cam = load_pose(
            cfg.dataset_root,
            scene,
            cfg.camera_type,
            ann_id,
            use_align=not cfg.no_align,
        )
        T_cam_world = np.linalg.inv(T_world_cam)

        depth_m = render_depth_from_mesh(
            ray_scene=ray_scene,
            K=K,
            T_cam_world=T_cam_world,
            width=width,
            height=height,
            depth_trunc=cfg.depth_trunc,
        )

        save_path = os.path.join(save_dir, f"{ann_id:04d}_depth.png")

        depth_raw = save_depth_png_from_meters(
            depth_m=depth_m,
            save_path=save_path,
            depth_scale=cfg.save_depth_scale,
        )

        # visualization every N frames
        if cfg.vis_every > 0 and (ann_id % cfg.vis_every == 0):
            save_depth_triplet_vis(
                cfg=cfg,
                scene=scene,
                camera_type=cfg.camera_type,
                ann_id=ann_id,
                K=K,
                factor_depth=factor_depth,
                fused_depth_m=depth_m,
                obs_depth_raw=sensor_raw,
            )

        # quick sanity check against original sensor depth
        sensor_m = sensor_raw.astype(np.float32) / float(factor_depth)
        render_m = depth_raw.astype(np.float32) * float(cfg.save_depth_scale) / 1000.0

        common = (sensor_m > 0) & (render_m > 0) & (sensor_m < cfg.depth_trunc)
        if common.sum() > 0:
            mae = np.mean(np.abs(sensor_m[common] - render_m[common]))
            valid_ratio = float((render_m > 0).mean())
            mae_list.append(mae)
            valid_ratio_list.append(valid_ratio)
        else:
            mae = np.nan
            valid_ratio = float((render_m > 0).mean())

        if (k + 1) % cfg.log_every == 0 or k == 0:
            print(
                f"[Render] {k + 1}/{len(frame_ids)} ann={ann_id:04d} "
                f"valid={valid_ratio:.4f}, sensor/render common MAE={mae:.4f} m"
            )

    if len(mae_list) > 0:
        print("[Summary]")
        print(f"  rendered frames: {len(frame_ids)}")
        print(f"  common-region MAE mean: {np.mean(mae_list):.4f} m")
        print(f"  common-region MAE median: {np.median(mae_list):.4f} m")
        print(f"  rendered valid ratio mean: {np.mean(valid_ratio_list):.4f}")


GRASPNET_SPLITS = {
    "train": list(range(0, 100)),
    "test": list(range(100, 190)),
    "test_seen": list(range(100, 130)),
    "test_similar": list(range(130, 160)),
    "test_novel": list(range(160, 190)),
}


def parse_scene_ids(scene_ids_str=None, scene_id=None, scene_start=None, scene_end=None):
    """
    Supported:
      --scene_id 160
      --scene_ids 160,161,162
      --scene_ids 160-189
      --scene_ids test_novel
      --scene_start 160 --scene_end 190   # end exclusive
    """
    if scene_ids_str is not None and len(scene_ids_str.strip()) > 0:
        s = scene_ids_str.strip()

        if s in GRASPNET_SPLITS:
            return GRASPNET_SPLITS[s]

        scene_ids = []
        for part in s.split(","):
            part = part.strip()
            if len(part) == 0:
                continue

            if "-" in part:
                a, b = part.split("-")
                a, b = int(a), int(b)
                if b >= a:
                    scene_ids.extend(list(range(a, b + 1)))
                else:
                    scene_ids.extend(list(range(a, b - 1, -1)))
            else:
                scene_ids.append(int(part))

        # keep order, remove duplicates
        scene_ids = list(dict.fromkeys(scene_ids))
        return scene_ids

    if scene_id is not None:
        return [int(scene_id)]

    if scene_start is not None and scene_end is not None:
        return list(range(int(scene_start), int(scene_end)))

    raise ValueError(
        "Please specify one of: --scene_id, --scene_ids, "
        "or --scene_start/--scene_end."
    )


def get_expected_done_path(cfg):
    scene = scene_name(cfg.scene_id)

    if cfg.skip_render:
        return os.path.join(
            cfg.output_root,
            "mesh",
            f"{scene}_{cfg.camera_type}_tsdf.ply",
        )

    frame_ids = list(range(cfg.start_ann, cfg.end_ann, cfg.render_stride))
    if len(frame_ids) == 0:
        return None

    last_ann = frame_ids[-1]
    return os.path.join(
        cfg.output_root,
        scene,
        cfg.camera_type,
        f"{last_ann:04d}_depth.png",
    )


def process_one_scene(scene_id, cfg):
    local_cfg = copy.copy(cfg)
    local_cfg.scene_id = int(scene_id)

    done_path = get_expected_done_path(local_cfg)
    if local_cfg.skip_done and done_path is not None and os.path.exists(done_path):
        return {
            "scene_id": scene_id,
            "ok": True,
            "skipped": True,
            "message": f"skip existing: {done_path}",
        }

    try:
        print(f"\n[Worker] start scene={scene_id:04d}")

        mesh, mesh_path = integrate_scene_tsdf(local_cfg)

        if not local_cfg.skip_render:
            render_scene_depths(local_cfg, mesh)

        print(f"[Worker] done scene={scene_id:04d}")

        return {
            "scene_id": scene_id,
            "ok": True,
            "skipped": False,
            "message": mesh_path,
        }

    except Exception:
        err = traceback.format_exc()
        print(f"[Worker][ERROR] scene={scene_id:04d}\n{err}")
        return {
            "scene_id": scene_id,
            "ok": False,
            "skipped": False,
            "message": err,
        }


def _worker_entry(args):
    scene_id, cfg = args
    return process_one_scene(scene_id, cfg)


def run_scenes(scene_ids, cfg):
    scene_ids = list(scene_ids)
    print(f"[Main] scenes={scene_ids}")
    print(f"[Main] proc={cfg.proc}")

    if cfg.proc <= 1:
        results = []
        for sid in scene_ids:
            results.append(process_one_scene(sid, cfg))
    else:
        ctx = mp.get_context(cfg.mp_start_method)

        # maxtasksperchild=1 helps release Open3D / mesh memory after each scene.
        with ctx.Pool(
            processes=int(cfg.proc),
            maxtasksperchild=1,
        ) as pool:
            jobs = [(sid, cfg) for sid in scene_ids]
            results = []
            for ret in pool.imap_unordered(_worker_entry, jobs):
                results.append(ret)
                status = "OK" if ret["ok"] else "FAIL"
                skipped = " skipped" if ret.get("skipped", False) else ""
                print(f"[Main] scene={ret['scene_id']:04d} {status}{skipped}: {ret['message']}")

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    skipped = [r for r in results if r.get("skipped", False)]

    print("\n[Main Summary]")
    print(f"  total:   {len(results)}")
    print(f"  ok:      {len(ok)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(fail)}")

    if len(fail) > 0:
        print("  failed scenes:", [r["scene_id"] for r in fail])
        
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", default="/data/robotarm/dataset/graspnet")
    parser.add_argument("--output_root", default="/data/robotarm/dataset/graspnet/tsdf_rendered_depth")
    parser.add_argument("--scene_id", type=int, default=None)
    parser.add_argument(
        "--scene_ids",
        default=None,
        help="Scene ids, e.g. '160', '160,161,162', '160-189', or split name: train/test/test_seen/test_similar/test_novel.",
    )
    parser.add_argument("--scene_start", type=int, default=None)
    parser.add_argument("--scene_end", type=int, default=None)

    parser.add_argument("--proc", type=int, default=1)
    parser.add_argument(
        "--mp_start_method",
        default="spawn",
        choices=["spawn", "forkserver", "fork"],
        help="Use spawn for better Open3D stability. If slow, try forkserver.",
    )

    parser.add_argument(
        "--skip_done",
        action="store_true",
        help="Skip a scene if the expected final output already exists.",
    )
    parser.add_argument("--camera_type", default="realsense", choices=["realsense", "kinect"])

    parser.add_argument(
    "--save_depth_scale",
    type=float,
    default=1.0,
    help="Save rendered depth as depth_m * 1000 / save_depth_scale. Use 1.0 for millimeter uint16 PNG.",
    )
    parser.add_argument("--start_ann", type=int, default=0)
    parser.add_argument("--end_ann", type=int, default=256)

    parser.add_argument("--integrate_stride", type=int, default=1)
    parser.add_argument("--render_stride", type=int, default=1)

    parser.add_argument("--voxel_length", type=float, default=0.004)
    parser.add_argument("--sdf_trunc", type=float, default=0.02)
    parser.add_argument("--depth_trunc", type=float, default=2.0)

    parser.add_argument("--volume_unit_resolution", type=int, default=16)
    parser.add_argument("--depth_sampling_stride", type=int, default=1)

    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--skip_render", action="store_true")
    parser.add_argument("--log_every", type=int, default=16)

    parser.add_argument(
        "--virtual_dataset_root",
        default=None,
        help="Path to virtual_scenes. If None, use dataset_root/virtual_scenes.",
    )

    parser.add_argument(
        "--vis_every",
        type=int,
        default=100,
        help="Save colored PLY visualization every N ann ids. <=0 disables visualization.",
    )

    parser.add_argument(
        "--vis_root",
        default="vis/tsdf_rendered_dpeth",
        help="Visualization root for colored PLY files.",
    )

    parser.add_argument(
        "--vis_max_points_per_cloud",
        type=int,
        default=100000,
        help="Randomly sample at most this many points per cloud in visualization. <=0 means no sampling.",
    )

    parser.add_argument(
        "--mesh_smooth",
        action="store_true",
        help="Apply mesh smoothing after TSDF extraction.",
    )

    parser.add_argument(
        "--mesh_smooth_iter",
        type=int,
        default=3,
        help="Number of Taubin smoothing iterations.",
    )

    cfg = parser.parse_args()
    scene_ids = parse_scene_ids(
        scene_ids_str=cfg.scene_ids,
        scene_id=cfg.scene_id,
        scene_start=cfg.scene_start,
        scene_end=cfg.scene_end,
    )

    run_scenes(scene_ids, cfg)


if __name__ == "__main__":
    main()