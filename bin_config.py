"""Read the names and sorting policy for the two physical bins."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "bins.json"


@dataclass(frozen=True)
class BinDefinition:
    id: str
    name: str
    rules: tuple[str, ...]


@dataclass(frozen=True)
class BinConfig:
    bins: tuple[BinDefinition, BinDefinition]
    fallback_bin: str
    hold_rules: tuple[str, ...]

    def get_bin(self, bin_id: str) -> BinDefinition:
        for definition in self.bins:
            if definition.id == bin_id:
                return definition
        raise ValueError(f"Unknown bin {bin_id!r}; use A or B.")

    def to_dict(self) -> dict:
        return {
            "bins": {
                definition.id: {"name": definition.name, "rules": list(definition.rules)}
                for definition in self.bins
            },
            "fallback_bin": self.fallback_bin,
            "hold_rules": list(self.hold_rules),
        }


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string.")
    return value.strip()


def _rules(value: object, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ValueError(f"{label} must be {'a' if allow_empty else 'a nonempty'} list of strings.")
    return tuple(_text(rule, f"{label}[{index}]") for index, rule in enumerate(value))


def _keys(value: object, expected: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must contain exactly these keys: {', '.join(sorted(expected))}.")
    return value


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        values[key] = value
    return values


def load_bin_config(path: Path = DEFAULT_CONFIG_PATH) -> BinConfig:
    """Fail early on incomplete/ambiguous config instead of guessing a policy."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        data = _keys(raw, {"bins", "fallback_bin", "hold_rules"}, "Configuration")
        bins = _keys(data["bins"], {"A", "B"}, "bins")
        definitions = []
        for bin_id in ("A", "B"):
            entry = _keys(bins[bin_id], {"name", "rules"}, f"bins.{bin_id}")
            definitions.append(BinDefinition(
                bin_id,
                _text(entry["name"], f"bins.{bin_id}.name"),
                _rules(entry["rules"], f"bins.{bin_id}.rules"),
            ))
        fallback = data["fallback_bin"]
        if fallback not in ("A", "B"):
            raise ValueError("fallback_bin must be A or B.")
        holds = _rules(data["hold_rules"], "hold_rules", allow_empty=True)
        return BinConfig((definitions[0], definitions[1]), fallback, holds)
    except (OSError, ValueError, UnicodeError) as exc:
        raise ValueError(f"Cannot load bin configuration {path}: {exc}") from exc
