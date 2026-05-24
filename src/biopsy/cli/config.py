"""TOML config loading and CLI option coalescing."""

from __future__ import annotations

import difflib
import tomllib
from math import isfinite
from pathlib import Path
from typing import Any

import typer

CONFIG_KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "target",
        "time",
        "time_col",
        "exclude",
        "exclude_file",
        "ignore_missing_exclude",
        "filter",
        "where",
        "sample",
        "target_sample",
        "shortlist",
        "cluster_cutoff",
        "html",
        "save",
        "plotly_cdn",
        "fast",
        "deep",
        "all_columns",
        "bins",
        "max_cols",
    }
)


def load_cli_config(path: Path | None, profile_name: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    expanded = path.expanduser()
    try:
        text = expanded.read_text()
    except OSError as exc:
        raise typer.BadParameter(f"Cannot read config {expanded}: {exc}") from exc
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise typer.BadParameter(f"Malformed TOML in {expanded}: {exc}") from exc
    profiles = data.get("profiles", {})
    cfg = {k: v for k, v in data.items() if k != "profiles"}
    check_config_keys(cfg, path, where="top-level")
    if profile_name:
        selected = profiles.get(profile_name)
        if selected is None:
            raise typer.BadParameter(f"Profile '{profile_name}' not found in {path}.")
        check_config_keys(selected, path, where=f"[profiles.{profile_name}]")
        cfg.update(selected)
    return cfg


def check_config_keys(cfg: dict[str, Any], path: Path, *, where: str) -> None:

    unknown = sorted(set(cfg) - CONFIG_KNOWN_KEYS)
    if not unknown:
        return
    suggestions = []
    for k in unknown:
        match = difflib.get_close_matches(k, sorted(CONFIG_KNOWN_KEYS), n=1, cutoff=0.6)
        if match:
            suggestions.append(f"'{k}' (did you mean '{match[0]}'?)")
        else:
            suggestions.append(f"'{k}'")
    raise typer.BadParameter(f"Unknown config key(s) in {path} {where}: {', '.join(suggestions)}.")


def fast_from_config(cfg: dict[str, Any]) -> Any:
    """Resolve TOML `fast` / `deep` aliases into the internal fast flag."""
    fast = bool_config_value(cfg, "fast")
    deep = bool_config_value(cfg, "deep")
    has_fast = fast is not None
    has_deep = deep is not None
    if has_fast and has_deep:
        if fast == deep:
            raise typer.BadParameter(
                "Config keys 'fast' and 'deep' conflict. Use one key, or set "
                "`fast = false` with `deep = true`."
            )
        return fast
    if has_fast:
        return cfg["fast"]
    if has_deep:
        return not bool(cfg["deep"])
    return None


def bool_config_value(cfg: dict[str, Any], key: str) -> bool | None:
    value = cfg.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise typer.BadParameter(f"Config key '{key}' must be true or false.")
    return value


def bool_option(name: str, value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise typer.BadParameter(f"Config key '{name}' must be true or false.")
    return value


def int_option(
    name: str,
    value: Any,
    *,
    default: int | None,
    min_value: int | None = None,
) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise typer.BadParameter(f"Config key '{name}' must be an integer.")
    if min_value is not None and value < min_value:
        raise typer.BadParameter(f"Config key '{name}' must be >= {min_value}.")
    return value


def float_option(
    name: str,
    value: Any,
    *,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise typer.BadParameter(f"Config key '{name}' must be a number.")
    coerced = float(value)
    if not isfinite(coerced):
        raise typer.BadParameter(f"Config key '{name}' must be finite.")
    if min_value is not None and coerced < min_value:
        raise typer.BadParameter(f"Config key '{name}' must be >= {min_value}.")
    if max_value is not None and coerced > max_value:
        raise typer.BadParameter(f"Config key '{name}' must be <= {max_value}.")
    return coerced


def coalesce(cli_value: Any, config_value: Any, *aliases: Any) -> Any:
    if config_value is None:
        for alias in aliases:
            if alias is not None:
                config_value = alias
                break
    return cli_value if cli_value not in (None, [], {}) else config_value


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def path_or_none(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def read_exclude_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    expanded = path.expanduser()
    try:
        text = expanded.read_text()
    except OSError as exc:
        raise typer.BadParameter(f"Cannot read exclude file {expanded}: {exc}") from exc
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            out.append(stripped)
    return out
