from typing import Sequence

import torch
from torch import nn


class SwinTransformer(nn.Module):
    """A lightweight Swin-Tiny style stub for debugging pipelines.

    The implementation does not depend on external configs and emits three
    feature levels whose channel counts follow the provided ``embed_dims`` and
    ``depths`` settings.
    """

    def __init__(
        self,
        embed_dims: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] | None = None,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        out_indices: Sequence[int] = (1, 2, 3),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        qkv_bias: bool = True,
        qk_scale=None,
        with_cp: bool = False,
        convert_weights: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.out_indices = out_indices
        channels = embed_dims
        self.stages = nn.ModuleList()
        for stage, depth in enumerate(depths):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                    )
                )
            downsample = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
            self.stages.append(nn.Sequential(*blocks, downsample))
            channels *= 2

    def forward(self, x: torch.Tensor):
        feats = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                feats.append(x)
        return feats
