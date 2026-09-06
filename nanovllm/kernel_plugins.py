"""Discovery and selection of optional external kernel plugins."""
from __future__ import annotations

from importlib.metadata import entry_points

_GROUP = "nanovllm.kernel_plugins"
_INSTANCES = {}


def load_kernel_plugin(name: str, operation: str):
    if not name:
        return None
    if name not in _INSTANCES:
        matches = [ep for ep in entry_points(group=_GROUP) if ep.name == name]
        if not matches:
            available = sorted(ep.name for ep in entry_points(group=_GROUP))
            raise RuntimeError(f"kernel plugin {name!r} not found; available: {available}")
        _INSTANCES[name] = matches[0].load()()
    plugin = _INSTANCES[name]
    if plugin.operation != operation:
        raise RuntimeError(
            f"kernel plugin {name!r} implements {plugin.operation!r}, not {operation!r}"
        )
    return plugin
