"""Model wiring for MIBiG built on clip_core primitives."""

from .classification import BGCClassifier
from .clip_dual import DualEncoderCLIP
from .projection import ProjectionHead

__all__ = ["BGCClassifier", "DualEncoderCLIP", "ProjectionHead"]
