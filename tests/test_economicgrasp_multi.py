"""Quick smoke-test for the BIP3D-based multimodal EconomicGrasp model."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from models.economicgrasp_multi import economicgrasp_multi_bip3d


class DummyBatch(Dataset):
    def __init__(self, num_points: int = 512, image_size: int = 224, length: int = 1):
        self.num_points = num_points
        self.image_size = image_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H = W = self.image_size
        N = self.num_points
        sample = {
            "point_clouds": torch.randn(N, 3),
            "imgs": torch.randn(3, H, W),
            "depths": torch.randn(1, H, W),
            "img_idxs": torch.randint(0, H * W, (N,)),
        }
        return sample


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def run_forward():
    dataset = DummyBatch()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    model = economicgrasp_multi_bip3d(is_training=False)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            summary = {k: v.shape if hasattr(v, "shape") else type(v) for k, v in outputs.items() if isinstance(v, torch.Tensor)}
            print("Forward pass keys:", sorted(summary.keys()))
            break


if __name__ == "__main__":
    run_forward()
