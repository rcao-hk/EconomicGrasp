import torch
from torch import nn
from torch.nn import functional as F
import models.extractors as extractors


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PSPNet(nn.Module):
    def __init__(
        self, n_classes=22, sizes=(1, 2, 3, 6), psp_size=2048,
        deep_features_size=1024, backend='resnet18', pretrained=True,
        out_dim=None,                 # <- 新增：统一输出维度（img_feat_dim）
        return_pyramid=False,         # <- 可选：默认不返回 pyramid（兼容旧代码）
    ):
        super().__init__()
        self.return_pyramid_default = return_pyramid
        self.out_dim = out_dim

        self.feats = getattr(extractors, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        # upsample path（保持你原逻辑）
        if deep_features_size == 64:
            self.up_1 = PSPUpsample(1024, 256)   # stride/8 -> /4
            self.up_2 = PSPUpsample(256, 64)     # /4 -> /2
            self.up_3 = PSPUpsample(64, 64)      # /2 -> /1
            c_p4, c_p2, c_p1 = 256, 64, 64
        elif deep_features_size == 128:
            self.up_1 = PSPUpsample(1024, 512)
            self.up_2 = PSPUpsample(512, 256)
            self.up_3 = PSPUpsample(256, 128)
            c_p4, c_p2, c_p1 = 512, 256, 128
        elif deep_features_size == 256:
            self.up_1 = PSPUpsample(1024, 512)
            self.up_2 = PSPUpsample(512, 256)
            self.up_3 = PSPUpsample(256, 256)
            c_p4, c_p2, c_p1 = 512, 256, 256
        else:
            raise ValueError(f"Unsupported deep_features_size={deep_features_size}")

        self.drop_2 = nn.Dropout2d(p=0.15)

        # -------- 新增：把每个尺度投影到 out_dim（img_feat_dim） --------
        # p8: psp 输出 1024 通道
        if out_dim is not None:
            self.proj_p8 = nn.Conv2d(1024, out_dim, kernel_size=1, bias=False)
            self.proj_p4 = nn.Conv2d(c_p4, out_dim, kernel_size=1, bias=False)
            self.proj_p2 = nn.Conv2d(c_p2, out_dim, kernel_size=1, bias=False)
            self.proj_p1 = nn.Conv2d(c_p1, out_dim, kernel_size=1, bias=False)
            self.proj_bn8 = nn.BatchNorm2d(out_dim)
            self.proj_bn4 = nn.BatchNorm2d(out_dim)
            self.proj_bn2 = nn.BatchNorm2d(out_dim)
            self.proj_bn1 = nn.BatchNorm2d(out_dim)
            self.proj_relu = nn.ReLU(inplace=True)
        else:
            self.proj_p8 = self.proj_p4 = self.proj_p2 = self.proj_p1 = None

    def _proj(self, x, proj, bn):
        x = proj(x)
        x = bn(x)
        return self.proj_relu(x)

    def forward(self, x, return_pyramid=None):
        if return_pyramid is None:
            return_pyramid = self.return_pyramid_default

        f, _ = self.feats(x)           # f: stride/8 (在你这个 resnet34 里)
        p8 = self.psp(f)               # (B,1024,H/8,W/8)
        p8 = self.drop_1(p8)

        p4 = self.up_1(p8)             # (B,*,H/4,W/4)
        p4 = self.drop_2(p4)

        p2 = self.up_2(p4)             # (B,*,H/2,W/2)
        p2 = self.drop_2(p2)

        p1 = self.up_3(p2)             # (B,*,H,W)

        if not return_pyramid:
            # 兼容旧逻辑：返回最终 p1（或你原本认为的 img_feat）
            return p1

        # 统一通道
        if self.out_dim is not None:
            p8o = self._proj(p8, self.proj_p8, self.proj_bn8)
            p4o = self._proj(p4, self.proj_p4, self.proj_bn4)
            p2o = self._proj(p2, self.proj_p2, self.proj_bn2)
            p1o = self._proj(p1, self.proj_p1, self.proj_bn1)
        else:
            p8o, p4o, p2o, p1o = p8, p4, p2, p1

        # p16：由 p8 再下采样一次（DeepViewAgg 常用最深层）
        p16o = F.avg_pool2d(p8o, kernel_size=2, stride=2)  # (B,C,H/16,W/16)

        return {"p1": p1o, "p2": p2o, "p4": p4o, "p8": p8o, "p16": p16o}