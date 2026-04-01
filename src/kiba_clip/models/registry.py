from __future__ import annotations

from typing import Any, Callable

from kiba_clip.models.clip_dual import DualEncoderCLIP

ModelBuilder = Callable[..., object]


class Registry:
    """Minimal name -> builder registry for extensibility."""

    def __init__(self) -> None:
        self._items: dict[str, ModelBuilder] = {}

    def register(self, name: str, builder: ModelBuilder) -> None:
        if name in self._items:
            raise ValueError(f"Registry item already exists: {name}")
        self._items[name] = builder

    def build(self, name: str, **kwargs: Any) -> object:
        if name not in self._items:
            raise KeyError(f"Unknown registry item: {name}")
        return self._items[name](**kwargs)


MODEL_REGISTRY = Registry()
MODEL_REGISTRY.register("dual_encoder_clip", DualEncoderCLIP)
