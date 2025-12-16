"""Self-contained BIP3D modules for multimodal feature extraction."""

from .swin import SwinTransformer
from .resnet import ResNet
from .channel_mapper import ChannelMapper

__all__ = ["SwinTransformer", "ResNet", "ChannelMapper"]
