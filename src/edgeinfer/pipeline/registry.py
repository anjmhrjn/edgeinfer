"""Model registry (D-037): published versions per model and the aliases that point at them.

`artifacts/registry/<model>.json` holds metadata only (ids, hashes, locations, report
paths), so it is committed like the reports. A version is a deployed artifact that passed
every gate; publishing the same artifact again reuses its version. An alias (`production`,
`dev`) names the version a consumer should load, and its history records every move.

Rollback moves an alias back to the version it held before the current one, skipping
versions already rolled back from, after checking that the artifact is still there with
the sha256 it was published with.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from edgeinfer.reports import REPO_ROOT

REGISTRY_DIR = REPO_ROOT / "artifacts" / "registry"
ALIASES = ("production", "dev")
INSTANCE_PREFIX = "instance:"  # location of a file that lives on the GPU instance

# Returns the sha256 of an artifact location, or None if it isn't there.
ShaLookup = Callable[[str], str | None]


class RegistryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _path(model: str, root: Path) -> Path:
    return root / f"{model}.json"


def load(model: str, root: Path = REGISTRY_DIR) -> dict[str, Any]:
    path = _path(model, root)
    if not path.is_file():
        return {"model": model, "versions": [], "aliases": {}, "history": []}
    return json.loads(path.read_text())


def _save(index: dict[str, Any], root: Path) -> Path:
    path = _path(index["model"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2) + "\n")
    return path


def _move(index: dict[str, Any], alias: str, version: int, action: str, reason: str) -> None:
    index["history"].append(
        {
            "at": _now(),
            "alias": alias,
            "action": action,
            "from": index["aliases"].get(alias),
            "to": version,
            "reason": reason,
        }
    )
    index["aliases"][alias] = version


def publish(record: dict[str, Any], alias: str, root: Path = REGISTRY_DIR) -> tuple[int, bool]:
    """Register a gated artifact and point `alias` at it. Returns (version, new): an
    artifact already registered (same id, file and target) keeps its version."""
    if alias not in ALIASES:
        raise RegistryError("REGISTRY_UNKNOWN_ALIAS", f"alias {alias!r} not in {list(ALIASES)}")
    index = load(record["model"], root)
    same = [
        v
        for v in index["versions"]
        if (v["artifact_id"], v["sha256"], v["target"])
        == (record["artifact_id"], record["sha256"], record["target"])
    ]
    if same:
        version, new = same[0]["version"], False
        # Publishing a version again promotes it on purpose, so it can be a rollback target.
        same[0]["rolled_back_from"] = [a for a in same[0]["rolled_back_from"] if a != alias]
    else:
        version, new = len(index["versions"]) + 1, True
        index["versions"].append(
            {"version": version, **record, "published_at": _now(), "rolled_back_from": []}
        )
    if index["aliases"].get(alias) != version:
        _move(index, alias, version, "publish", f"deploy run {record['run_id']}")
    _save(index, root)
    return version, new


def verify(version: dict[str, Any], sha_of: ShaLookup) -> None:
    """The version's file must still exist with the sha256 it was published with."""
    found = sha_of(version["location"])
    if found is None:
        raise RegistryError(
            "REGISTRY_ARTIFACT_MISSING",
            f"v{version['version']}: {version['location']} doesn't exist "
            "(an engine is lost when the instance is terminated; rebuild and "
            "deploy again)",
        )
    if found != version["sha256"]:
        raise RegistryError(
            "REGISTRY_ARTIFACT_MISSING",
            f"v{version['version']}: {version['location']} has sha256 "
            f"{found[:12]}, published as {version['sha256'][:12]}; it was "
            "replaced (a rebuild gets new bytes and must pass the gates again)",
        )


def rollback_target(index: dict[str, Any], alias: str) -> int:
    """The version `alias` held before its current one, skipping rolled-back versions."""
    current = index["aliases"].get(alias)
    if current is None:
        raise RegistryError("REGISTRY_NO_PREVIOUS", f"alias {alias!r} points at no version")
    by_version = {v["version"]: v for v in index["versions"]}
    held = [h["to"] for h in index["history"] if h["alias"] == alias]
    for version in reversed(held):
        if version != current and alias not in by_version[version]["rolled_back_from"]:
            return version
    raise RegistryError(
        "REGISTRY_NO_PREVIOUS", f"alias {alias!r} (v{current}) has no earlier version to go back to"
    )


def rollback(
    model: str, alias: str, sha_of: ShaLookup, reason: str, root: Path = REGISTRY_DIR
) -> tuple[int, int]:
    """Move `alias` back one version. Returns (from, to)."""
    index = load(model, root)
    target = rollback_target(index, alias)
    by_version = {v["version"]: v for v in index["versions"]}
    verify(by_version[target], sha_of)
    current = index["aliases"][alias]
    by_version[current]["rolled_back_from"].append(alias)
    _move(index, alias, target, "rollback", reason)
    _save(index, root)
    return current, target


def local_sha(location: str) -> str | None:
    """sha256 of a location on this machine (paths relative to the repo root)."""
    from edgeinfer.artifacts import sha256_file

    if location.startswith(INSTANCE_PREFIX):
        raise RegistryError(
            "REGISTRY_ARTIFACT_MISSING",
            f"{location} is on the GPU instance; check it with the remote lookup",
        )
    path = Path(location)
    path = path if path.is_absolute() else REPO_ROOT / path
    return sha256_file(path) if path.is_file() else None
