import torch
from torch import nn


def _make_norm(norm_cfg, num_channels: int):
    if norm_cfg is None:
        return None
    if norm_cfg.get("type") == nn.GroupNorm:
        return nn.GroupNorm(norm_cfg.get("num_groups", 32), num_channels)
    if norm_cfg.get("type") == "GN":
        return nn.GroupNorm(norm_cfg.get("num_groups", 32), num_channels)
    return None


class ChannelMapper(nn.Module):
    """Project multi-scale features to a unified channel width."""

    def __init__(
        self,
        in_channels,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = True,
        act_cfg=None,
        norm_cfg=None,
        num_outs: int | None = None,
    ):
        super().__init__()
        num_outs = num_outs or len(in_channels)
        self.convs = nn.ModuleList()
        for c in in_channels:
            layers = [
                nn.Conv2d(c, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=bias)
            ]
            norm = _make_norm(norm_cfg or {}, out_channels)
            if norm is not None:
                layers.append(norm)
            if act_cfg:
                layers.append(nn.ReLU(inplace=True))
            self.convs.append(nn.Sequential(*layers))
        self.num_outs = num_outs

    def forward(self, feats):
        outs = [conv(x) for conv, x in zip(self.convs, feats)]
        # pad extra levels by downsampling the last map if needed
        while len(outs) < self.num_outs:
            outs.append(nn.functional.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return outs[: self.num_outs]
