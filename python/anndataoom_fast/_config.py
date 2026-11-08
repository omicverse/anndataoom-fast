"""Runtime switch: ``ANNDATAOOM_FAST=0`` (or :func:`disable`) forces the
original anndataoom code paths."""
import os

_state = {"enabled": os.environ.get("ANNDATAOOM_FAST", "1").lower() not in ("0", "false", "no", "off")}


def enabled() -> bool:
    return _state["enabled"]


def enable() -> None:
    _state["enabled"] = True


def disable() -> None:
    _state["enabled"] = False
