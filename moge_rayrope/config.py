"""Explicit experiment configuration; defaults do not enable new components."""
from dataclasses import dataclass
import math

MAIN_COMMIT = "52d09f925059bec3643610ecf1f1722894627ee5"
BRANCH = "exp/moge-rayrope-grasp20"
VERSION = "economicgrasp_moge_rayrope_v1"

@dataclass(frozen=True)
class ModelConfig:
    encoder: str = "vitb"
    use_moge: bool = False
    use_rayrope: bool = False
    ray_encoding: str = "expected"  # none: same attention without RoPE
    uncertainty: str = "fixed"      # point/none do not execute a sigma head
    uncertainty_loss: str = "interval"  # interval / laplace_decoupled / laplace_joint
    fixed_halfwidth: float = 0.02
    sigma_min: float = 0.001
    sigma_max: float = 0.08
    interval_coverage: float = 0.9
    pose_mode: str = "global_film"
    min_depth: float = 0.2
    max_depth: float = 1.0
    seeds: int = 1024
    group_chunk: int = 64
    ray_grid: int = 7
    ray_radius_px: float = 40.0
    ray_dim: int = 192
    ray_heads: int = 4
    ray_freq_base: float = 2.0
    ray_apply_vo: bool = True
    virtual_standoff: float = 0.15
    ray_length_unit: float = 0.1
    use_shape_tokens: bool = False
    shape_stride: int = 4
    calibration_detach: bool = True
    checkpoint_chunks: bool = True

    def __post_init__(self):
        if self.encoder not in ("vits", "vitb", "vitl"):
            raise ValueError("encoder must be vits/vitb/vitl")
        if self.pose_mode not in ("none", "global_film"):
            raise ValueError("This controlled experiment supports none/global_film")
        if self.ray_encoding not in ("none", "point", "expected"):
            raise ValueError("ray_encoding must be none/point/expected")
        if self.uncertainty not in ("fixed", "learned"):
            raise ValueError("uncertainty must be fixed/learned")
        if self.uncertainty_loss not in ("interval", "laplace_decoupled", "laplace_joint"):
            raise ValueError("unknown uncertainty_loss")
        if self.uncertainty_loss != "interval" and not (
            self.use_rayrope and self.ray_encoding == "expected" and self.uncertainty == "learned"
        ):
            raise ValueError("Laplace confidence loss requires expected RayRoPE + learned sigma")
        numeric = (self.min_depth,self.max_depth,self.fixed_halfwidth,
                   self.sigma_min,self.sigma_max,self.interval_coverage,
                   self.ray_radius_px,self.ray_freq_base,
                   self.virtual_standoff,self.ray_length_unit)
        if not all(math.isfinite(x) for x in numeric):
            raise ValueError("Nonfinite model configuration")
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("Invalid metric depth interval")
        if not 0 < self.sigma_min < self.fixed_halfwidth < self.sigma_max:
            raise ValueError("Need 0 < sigma_min < fixed_halfwidth < sigma_max")
        if not 0 < self.interval_coverage < 1:
            raise ValueError("interval_coverage must be in (0,1)")
        if min(self.seeds,self.group_chunk,self.ray_grid,self.ray_dim,self.ray_heads,self.shape_stride) < 1:
            raise ValueError("Counts must be positive")
        if self.ray_grid % 2 != 1 or self.ray_grid < 3:
            raise ValueError("Use an odd ray_grid >=3")
        if self.ray_dim % self.ray_heads or (self.ray_dim//self.ray_heads) % 12:
            raise ValueError("Each head needs 6 coordinates x sin/cos x integer frequencies")
        if min(self.ray_radius_px,self.virtual_standoff,self.ray_length_unit) <= 0 or self.ray_freq_base < 1:
            raise ValueError("Invalid ray geometry/frequency configuration")
        if not self.calibration_detach:
            raise ValueError("This experiment fixes the shape/metric gradient boundary")
        if self.use_shape_tokens and not (self.use_moge and self.use_rayrope):
            raise ValueError("shape_tokens requires both MoGe and RayRoPE")
        if self.uncertainty == "learned" and not (self.use_rayrope and self.ray_encoding == "expected"):
            raise ValueError("learned uncertainty requires expected RayRoPE")

@dataclass(frozen=True)
class LossConfig:
    global_shape: float = 1.0
    local_shape: float = 0.5
    reprojection: float = 0.05
    interval: float = 0.1
    global_points: int = 128
    local_points: int = 48
    patches_per_scale: int = 2
    patch_sizes: tuple = (8,16,32)

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in
               (self.global_shape,self.local_shape,self.reprojection,self.interval)):
            raise ValueError("Loss weights must be finite nonnegative")
        if min(self.global_points,self.local_points,self.patches_per_scale) < 2:
            raise ValueError("Too few alignment points/patches")
        if not self.patch_sizes or any(int(p)<2 for p in self.patch_sizes):
            raise ValueError("Invalid patch sizes")
