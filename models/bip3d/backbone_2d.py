"""Lightweight 2D backbone implementations for BIP3D."""
import torch
from torch import nn


class ConvBackbone2D(nn.Module):
    """A tiny CNN backbone that mimics multi-level feature extraction.

    The module stacks a configurable number of convolutional blocks and
    returns intermediate feature maps so downstream FPN-style necks can reuse
    them. It is intentionally simple to keep the dependency surface small
    while providing the spatial output shapes expected by the decoder.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_stages: int = 3):
        super().__init__()
        layers = []
        channels = in_channels
        for stage in range(num_stages):
            out_channels = base_channels * (2 ** stage)
            layers.append(
                nn.Sequential(
                    nn.Conv2d(channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            channels = out_channels
        self.stages = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor):
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class IdentityBackbone2D(nn.Module):
    """A placeholder backbone that simply wraps the input in a list."""

    def forward(self, x: torch.Tensor):
        return [x]
