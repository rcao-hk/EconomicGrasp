from typing import Sequence

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = torch.relu(out + identity)
        return out


class ResNet(nn.Module):
    """Minimal ResNet-34 style encoder returning feature pyramid."""

    def __init__(
        self,
        depth: int = 34,
        in_channels: int = 1,
        base_channels: int = 64,
        num_stages: int = 4,
        out_indices: Sequence[int] = (1, 2, 3),
        with_cp: bool = False,
        bn_eval: bool = False,
        style: str = "pytorch",
        **kwargs,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        channels = base_channels
        self.layers = nn.ModuleList()
        for stage in range(num_stages):
            stride = 2 if stage > 0 else 1
            layer = [BasicBlock(channels, channels, stride=stride)]
            layer.append(BasicBlock(channels, channels))
            self.layers.append(nn.Sequential(*layer))
            channels = channels
            if stage < num_stages - 1:
                channels *= 2

    def forward(self, x: torch.Tensor):
        feats = []
        x = self.stem(x)
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in self.out_indices:
                feats.append(x)
        return feats
