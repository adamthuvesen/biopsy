"""JSON-safe serialization helpers for public report objects."""

from __future__ import annotations

import math
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Strings that look like a URI with userinfo: scheme://user:pass@host or
# scheme://user@host. The adapter contract strips these before we ever
# reach serialization, but this catches mistakes — defense in depth so a
# leaked credential never survives a save+share round-trip.
_USERINFO_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<userinfo>[^/@\s]+@)")


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and common scientific Python scalars to plain data."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, float):
        # NaN and ±Inf both collapse to None — JSON has no native non-finite
        # encoding. Round-tripped profiles cannot distinguish "all-null mean"
        # from "overflow", so callers shouldn't rely on Inf surviving save/load.
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _redact_userinfo(value)
    return value


def _redact_userinfo(value: str) -> str:
    """Strip `user:pass@` from any URI-shaped substring.

    Adapter code already produces credential-free `qualified` URIs, so this
    is belt-and-braces: a stray `source_uri` set from raw user input still
    can't carry credentials into a saved profile.
    """
    if "://" not in value:
        return value
    return _USERINFO_RE.sub(r"\g<scheme>", value)
