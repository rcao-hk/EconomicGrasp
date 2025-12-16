import torch
from torch import nn

from ..utils.build import build


def _maybe_build(cfg, *args):
    if cfg is None:
        return None
    return build(cfg, *args)


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
        conv_cfg=None,
    ):
        super().__init__()
        num_outs = num_outs or len(in_channels)
        self.convs = nn.ModuleList()
        for c in in_channels:
            conv = _maybe_build(
                conv_cfg
                or dict(
                    type=nn.Conv2d,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=bias,
                ),
                c,
                out_channels,
            )
            layers = [conv]
            norm = _maybe_build(norm_cfg or {}, out_channels)
            if norm is not None:
                layers.append(norm)
            if act_cfg:
                act = _maybe_build(act_cfg)
                layers.append(act if act is not None else nn.ReLU(inplace=True))
            self.convs.append(nn.Sequential(*layers))
        self.num_outs = num_outs

    def forward(self, feats):
        outs = [conv(x) for conv, x in zip(self.convs, feats)]
        # pad extra levels by downsampling the last map if needed
        while len(outs) < self.num_outs:
            outs.append(nn.functional.max_pool2d(outs[-1], kernel_size=1, stride=2))
        return outs[: self.num_outs]
