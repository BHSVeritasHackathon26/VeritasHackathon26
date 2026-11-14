"""Loads a participant config file (see configs/t15.toml)."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ParticipantConfig:
    participant: str
    data_dir: Path
    splits_file: Path
    n_features: int
    bin_ms: int
    tune_fraction: float
    seed: int
    holdout_sessions: tuple
    files: dict  # official split name -> file name inside a session folder


def load_config(path):
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    def resolve(p):
        # Paths in the config are relative to the repo, so it works on any machine.
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    return ParticipantConfig(
        participant=raw["participant"],
        data_dir=resolve(raw["data_dir"]),
        splits_file=resolve(raw["splits_file"]),
        n_features=raw["n_features"],
        bin_ms=raw["bin_ms"],
        tune_fraction=raw["tune_fraction"],
        seed=raw["seed"],
        holdout_sessions=tuple(raw["holdout_sessions"]),
        files=dict(raw["files"]),
    )
