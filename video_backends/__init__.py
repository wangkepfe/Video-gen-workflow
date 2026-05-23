"""Video-generation backend registry.

A backend is a Python module that follows the contract documented in
`video_backends/base.py`. New backends are registered by importing them
here and adding them to BACKENDS.
"""
import os
from . import wan22, ltx
from .base import Field, MissingFiles

BACKENDS = {b.NAME: b for b in (wan22, ltx)}
DEFAULT_BACKEND = wan22.NAME


def get(name):
    """Look up a backend module by NAME. Raises KeyError with a helpful message."""
    if name not in BACKENDS:
        raise KeyError(
            f"Unknown video backend: {name!r}. "
            f"Available: {', '.join(BACKENDS.keys())}"
        )
    return BACKENDS[name]


def list_backends():
    """Return a list of dicts suitable for JSON serialisation to the web UI."""
    out = []
    for name, mod in BACKENDS.items():
        out.append({
            "name": name,
            "display_name": mod.DISPLAY_NAME,
            "description": getattr(mod, "DESCRIPTION", ""),
            "default_negative": getattr(mod, "DEFAULT_NEGATIVE", ""),
            "fields": [f.to_dict() for f in mod.FIELDS],
            "required_files_present": all(os.path.isfile(p) for _, p in mod.required_files()),
        })
    return out


def field_defaults(name):
    """Return {field_id: default_value} for a backend."""
    return {f.id: f.default for f in get(name).FIELDS if f.default is not None}


__all__ = ["BACKENDS", "DEFAULT_BACKEND", "Field", "MissingFiles",
           "get", "list_backends", "field_defaults"]
