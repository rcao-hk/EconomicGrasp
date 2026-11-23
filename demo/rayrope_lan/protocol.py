"""Length-prefixed NPZ transport used by the LAN grasp demo.

The protocol deliberately avoids pickle: arrays are serialized with NumPy's
NPZ format and metadata is UTF-8 JSON stored as a uint8 array.
"""
from __future__ import annotations

import io
import socket
import struct
from typing import Mapping

import numpy as np


_HEADER = struct.Struct("!Q")


def encode_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def decode_npz(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def recv_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise ValueError(f"Invalid receive size: {size}")
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(
                f"Connection closed with {remaining} bytes still expected."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(
    sock: socket.socket,
    max_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    header = recv_exact(sock, _HEADER.size)
    (payload_size,) = _HEADER.unpack(header)
    if payload_size <= 0:
        raise ValueError(f"Empty frame is not valid (size={payload_size}).")
    if payload_size > int(max_bytes):
        raise ValueError(
            f"Incoming frame is too large: {payload_size} bytes "
            f"(limit={int(max_bytes)})."
        )
    return recv_exact(sock, int(payload_size))


def send_frame(sock: socket.socket, payload: bytes) -> None:
    if not payload:
        raise ValueError("Cannot send an empty frame.")
    sock.sendall(_HEADER.pack(len(payload)) + payload)
