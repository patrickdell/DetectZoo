"""Global detector registry and factory function."""

from __future__ import annotations

from typing import Any, Type

from detectzoo.core.base import BaseDetector

_REGISTRY: dict[str, Type[BaseDetector]] = {}
_ALIASES: dict[str, str] = {}


def register_detector(name: str, aliases: list[str] | None = None):
    """Class decorator that registers a detector under *name*.

    Parameters:
        name: Primary registration name.
        aliases: Optional list of alternative names that resolve to the
            same detector class.

    Usage::

        @register_detector("my_detector", aliases=["my_det"])
        class MyDetector(BaseDetector):
            ...
    """

    def decorator(cls: Type[BaseDetector]) -> Type[BaseDetector]:
        if name in _REGISTRY:
            raise ValueError(
                f"Detector '{name}' is already registered ({_REGISTRY[name].__name__}). "
                "Use a different name."
            )
        cls.name = name
        _REGISTRY[name] = cls
        for alias in aliases or []:
            _ALIASES[alias] = name
        return cls

    return decorator


def load_detector(name: str, **kwargs: Any) -> BaseDetector:
    """Instantiate a registered detector by name.

    Parameters:
        name: Registered detector identifier (e.g. ``"detectgpt"``).
        **kwargs: Forwarded to the detector constructor.

    Returns:
        An initialised detector instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    resolved = _ALIASES.get(name, name)
    if resolved not in _REGISTRY:
        available = ", ".join(sorted(set(_REGISTRY) | set(_ALIASES))) or "(none)"
        raise ValueError(
            f"Unknown detector '{name}'. Available detectors: {available}"
        )
    return _REGISTRY[resolved](**kwargs)


def list_detectors(modality: str | None = None) -> list[str]:
    """Return names of all registered detectors, optionally filtered by modality."""
    if modality is None:
        return sorted(_REGISTRY)
    return sorted(
        name for name, cls in _REGISTRY.items() if cls.modality == modality
    )
