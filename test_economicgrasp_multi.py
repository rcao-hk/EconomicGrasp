"""Quick smoke-test for the BIP3D-based multimodal EconomicGrasp model."""

from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from models.economicgrasp_multi import economicgrasp_multi_bip3d
# from models.economicgrasp import economicgrasp_multi
# from models.economicgrasp_2d import EconomicGrasp_ImageCenter
# from models.economicgrasp_depth_c1 import economicgrasp_c1
from models.economicgrasp_depth import EconomicGrasp_RGBDepthProb

class DummyBatch(Dataset):
    def __init__(self, num_points: int = 512, image_size: int = 448, length: int = 1):
        self.num_points = num_points
        self.image_size = image_size
        self.length = length
        self.voxel_size = 0.002
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        H = W = self.image_size
        N = self.num_points

        # 单视角：V=1
        V = 1

        # 让 depth 为正，避免 GT depth_prob 里全变 invalid
        depths = torch.rand(V, H, W) * 2.0 + 0.25  # [0.25, 2.25]

        sample = {
            "point_clouds": torch.randn(N, 3),                 # (N,3)
            "imgs": torch.randn(V, 3, H, W),                   # ✅ (V,3,H,W)
            "img": torch.randn(3, H, W),                   # ✅ (V,3,H,W)
            "depth": depths,                                   # ✅ (V,1,H,W)
            "img_idxs": torch.randint(0, H * W, (N,)),         # (N,) flatten idx in 224*224
            "coordinates_for_voxel": torch.randn(N, 3) / self.voxel_size,
            # ✅ spatial_enhancer 需要
            "image_wh": torch.tensor([W, H], dtype=torch.float32).unsqueeze(0),   # (1,2) or (2)都行
            "projection_mat": torch.eye(4, dtype=torch.float32).unsqueeze(0),     # ✅ (V,4,4)
            "K": torch.eye(3, dtype=torch.float32),
        }
        return sample

def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    # out["imgs"]:   (B,V,3,H,W)
    # out["depths"]: (B,V,1,H,W)
    # out["projection_mat"]: (B,V,4,4)
    return out

def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        y = [to_device(v, device) for v in x]
        return tuple(y) if isinstance(x, tuple) else y
    return x


def run_forward():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset = DummyBatch()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    # model = economicgrasp_multi_bip3d(is_training=False)
    # model = economicgrasp_multi(is_training=False)
    model = EconomicGrasp_RGBDepthProb(is_training=False)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(batch)
            summary = {k: v.shape if hasattr(v, "shape") else type(v) for k, v in outputs.items() if isinstance(v, torch.Tensor)}
            print("Forward pass keys:", sorted(summary.keys()))
            break


if __name__ == "__main__":
    run_forward()
