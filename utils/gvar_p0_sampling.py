"""Pure helpers for matched P0 Original/GN-Trans RGB inputs.

This module must not import argparse, legacy config, dataset, CUDA or graspnetAPI;
it is unit-testable without the real grasp training environment.
"""
from pathlib import Path

SEED_MODULUS = 2**32 - 1


def per_frame_sampling_seed(index: int, user_seed: int) -> int:
    """Deterministic across domains, loader workers, execution order."""
    if index < 0 or not 0 <= user_seed < 2**32:
        raise ValueError('invalid per-frame seed/index')
    return (user_seed * 2654435761 + (index + 1) * 2246822519) % SEED_MODULUS


def gntrans_rgb_path(rgb_root: Path, scene_name: str, frame_id: int) -> Path:
    if not (scene_name.startswith('scene_') and scene_name[6:].isdigit()
            and len(scene_name[6:]) == 4 and 0 <= frame_id < 256):
        raise ValueError(f'Invalid scene/frame: {scene_name}/{frame_id}')
    return Path(rgb_root) / 'scenes' / f'{int(scene_name[6:]):05d}' / f'{frame_id:04d}_color.png'
