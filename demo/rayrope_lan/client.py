"""LAN client for the Point RayRoPE grasp inference server.

Example:
    python -m demo.rayrope_lan.client \
        --server-ip 192.168.1.20 --port 8765 \
        --rgb /tmp/crop_rgb.png --depth /tmp/crop_depth.png \
        --metadata /tmp/crop_meta.json --output /tmp/top10_grasps.npz
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import numpy as np

from .protocol import decode_npz, encode_npz, recv_frame, send_frame


def _load_rgb(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_depth(path: str) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix.lower() == ".npy":
        depth = np.load(file_path, allow_pickle=False)
    else:
        from PIL import Image

        with Image.open(file_path) as image:
            depth = np.asarray(image)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Depth image must be single-channel, got {depth.shape}.")
    if depth.dtype.kind not in {"u", "i", "f"}:
        raise ValueError(f"Unsupported depth dtype: {depth.dtype}.")
    return depth


def _decode_metadata(response: dict[str, np.ndarray]) -> dict:
    value = response.get("metadata_json")
    if value is None:
        raise RuntimeError("Server response is missing metadata_json.")
    return json.loads(np.asarray(value, dtype=np.uint8).tobytes().decode("utf-8"))


def decode_response_metadata(response: dict[str, np.ndarray]) -> dict:
    """Decode the server's JSON metadata from a response NPZ."""
    return _decode_metadata(response)


def send_rgbd_request(
    server_ip: str,
    port: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    metadata: dict,
    timeout: float = 60.0,
    auth_token: str = "",
    max_response_bytes: int = 64 * 1024 * 1024,
) -> dict[str, np.ndarray]:
    """Send in-memory RGB/depth crops and return the decoded response arrays."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be uint8 [Hcrop,Wcrop,3].")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.shape != rgb.shape[:2]:
        raise ValueError("depth must be single-channel and aligned with rgb.")
    if depth.dtype.kind not in {"u", "i", "f"}:
        raise ValueError(f"Unsupported depth dtype: {depth.dtype}.")

    request_metadata = dict(metadata)
    request_metadata.setdefault("rgb_color_order", "RGB")
    if auth_token:
        request_metadata["auth_token"] = auth_token
    metadata_bytes = json.dumps(
        request_metadata, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request_payload = encode_npz({
        "rgb": rgb,
        "depth": depth,
        "metadata_json": np.frombuffer(metadata_bytes, dtype=np.uint8),
    })

    with socket.create_connection(
        (server_ip, int(port)), timeout=float(timeout)
    ) as connection:
        connection.settimeout(float(timeout))
        send_frame(connection, request_payload)
        response_payload = recv_frame(
            connection, max_bytes=int(max_response_bytes)
        )

    response = decode_npz(response_payload)
    response_metadata = _decode_metadata(response)
    if response_metadata.get("status") != "ok":
        raise RuntimeError(
            response_metadata.get("error", "Inference server returned an error.")
        )
    return response


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send one cropped RGB-D scene to the LAN grasp server.")
    parser.add_argument("--server-ip", required=True, help="GPU server's private-LAN IPv4/IPv6 address.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rgb", required=True, help="Raw-resolution cropped RGB image.")
    parser.add_argument("--depth", required=True, help="Aligned raw-resolution cropped depth PNG or NPY.")
    parser.add_argument("--metadata", required=True, help="JSON file containing crop box, full-image K and camera pose.")
    parser.add_argument("--depth-scale-to-m", type=float, default=None,
                        help="Optional override; uint16 millimeters=0.001, float meters=1.0.")
    parser.add_argument("--auth-token", default="", help="Optional shared LAN token.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-response-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--output", default="", help="Optional output NPZ path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rgb = _load_rgb(args.rgb)
    depth = _load_depth(args.depth)
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB/depth crops must align: rgb={rgb.shape[:2]}, depth={depth.shape}."
        )

    with open(args.metadata, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must contain an object.")
    if args.depth_scale_to_m is not None:
        metadata["depth_scale_to_m"] = float(args.depth_scale_to_m)
    if args.auth_token:
        metadata["auth_token"] = args.auth_token
    metadata.setdefault("rgb_color_order", "RGB")

    response = send_rgbd_request(
        server_ip=args.server_ip,
        port=args.port,
        rgb=rgb,
        depth=depth,
        metadata=metadata,
        timeout=args.timeout,
        max_response_bytes=args.max_response_bytes,
    )
    response_metadata = _decode_metadata(response)
    print(json.dumps(response_metadata, indent=2, ensure_ascii=False))

    if response_metadata.get("status") != "ok":
        raise RuntimeError(response_metadata.get("error", "Inference server returned an error."))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **response)
        print(f"Saved grasp response to: {output_path}")

    grasps = np.asarray(response.get("grasps_camera", np.empty((0, 17))))
    print(f"Returned {len(grasps)} scene-level grasps (camera frame).")
    for rank, grasp in enumerate(grasps, start=1):
        score = float(grasp[0])
        width = float(grasp[1])
        rotation = grasp[4:13].reshape(3, 3)
        translation = grasp[13:16]
        print(
            f"rank={rank:02d} score={score:.4f} width_m={width:.4f} "
            f"t_m={np.array2string(translation, precision=4)}"
        )
        print(np.array2string(rotation, precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
