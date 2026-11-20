#!/usr/bin/env python3
"""Generate a CycloneDX SBOM for the locked desktop dependencies.

The lock file is the source of truth: it is what CI installs with
``uv sync --locked``. Keeping the generator dependency-free means the release
job can run it without installing anything first.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Iterable, Optional

# Fixed namespace so repeated SBOMs for the same lock file are byte-identical.
SBOM_NAMESPACE = uuid.UUID("6f6d2c1e-6a1c-5c1e-9f3b-2b1b2d4c5e6f")


class SbomError(RuntimeError):
    """Raised when the lock file cannot be read."""


def _purl(name: str, version: str) -> str:
    return f"pkg:pypi/{name.lower().replace('_', '-')}@{version}"


def components_from_lock(lock_path: Path) -> list[dict]:
    try:
        with Path(lock_path).open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise SbomError(f"could not read {lock_path}: {exc}") from exc
    packages = data.get("package", [])
    if not isinstance(packages, list):
        raise SbomError(f"{lock_path} has no package list")
    components = []
    for package in packages:
        name = package.get("name")
        version = package.get("version")
        if not name or not version:
            continue
        components.append(
            {
                "type": "library",
                "name": str(name),
                "version": str(version),
                "purl": _purl(str(name), str(version)),
            }
        )
    components.sort(key=lambda item: (item["name"].lower(), item["version"]))
    return components


def build_sbom(components: list[dict], version: str) -> dict:
    digest = json.dumps(components, sort_keys=True).encode("utf-8")
    serial = uuid.uuid5(SBOM_NAMESPACE, digest.hex())
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "openvrcardboard",
                "version": version,
                "licenses": [{"license": {"id": "MIT"}}],
            }
        },
        "components": components,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--lock", type=Path, default=Path("desktop/uv.lock"), help="uv lock file"
    )
    parser.add_argument("--version", required=True, help="application version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        components = components_from_lock(args.lock)
    except SbomError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sbom = build_sbom(components, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n")
    print(f"{args.output}: {len(components)} components")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
