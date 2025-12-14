"""Simple 2D neck modules for BIP3D."""
from torch import nn


class IdentityNeck2D(nn.Module):
    """Return features as-is for pipelines that do not require a neck."""

    def forward(self, feats):
        return feats


class PyramidFusionNeck(nn.Module):
    """Lightweight FPN-style fusion.

    Each feature map is projected to the same channel dimension and then
    upsampled to the resolution of the finest scale before summation.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )

    def forward(self, feats):
        fused = None
        for feat, proj in zip(feats, self.lateral_convs):
            projected = proj(feat)
            if fused is None:
                fused = projected
            else:
                projected = nn.functional.interpolate(
                    projected,
                    size=fused.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                fused = fused + projected
        return [fused]
