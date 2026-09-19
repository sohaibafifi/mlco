#!/usr/bin/env python3
"""Check or update the workspace's shared release version (Python 3.11+)."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("aet", "attr", "cax", "cli", "core", "dual", "probe", "problems", "scale", "xai")
RUNTIME_MODULES = ("aet", "attr", "cax", "cli", "dual", "probe", "xai")
VERSION = r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
INTERNAL = re.compile(r"(neuro-co-[a-z]+)(\[[a-z0-9,-]+\])?(.*)")


def bounds(version: str) -> str:
    """Keep pre-1.0 dependencies within a minor release, then within a major."""
    if re.fullmatch(VERSION, version) is None:
        raise ValueError(f"Expected a stable X.Y.Z version, got {version!r}")
    major, minor, _ = map(int, version.split("."))
    upper = str(major + 1) if major else f"0.{minor + 1}"
    return f">={version},<{upper}"


def replace_once(text: str, pattern: str, replacement: str, path: Path) -> str:
    updated, count = re.subn(pattern, lambda _: replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected one version field in {path.relative_to(ROOT)}")
    return updated


def plan(target: str | None = None) -> tuple[str, dict[Path, str]]:
    """Validate every current field before returning any edits."""
    names = {f"neuro-co-{package}" for package in PACKAGES}
    manifests = sorted((ROOT / "packages").glob("*/pyproject.toml"))
    if {path.parent.name for path in manifests} != names:
        raise ValueError("The package list changed; update PACKAGES in scripts/version.py")
    current = tomllib.loads(manifests[0].read_text())["project"]["version"]
    expected_bounds = bounds(current)
    new_version = target or current
    new_bounds = bounds(new_version)
    edits = {}
    for path in manifests:
        text = path.read_text()
        project = tomllib.loads(text)["project"]
        if project["name"] != path.parent.name or project["version"] != current:
            raise ValueError(f"Inconsistent package name or version in {path.relative_to(ROOT)}")
        updated = replace_once(
            text, rf'^version = "{re.escape(current)}"$', f'version = "{new_version}"', path
        )
        dependencies = list(project.get("dependencies", []))
        for optional in project.get("optional-dependencies", {}).values():
            dependencies.extend(optional)
        for dependency in dependencies:
            match = INTERNAL.fullmatch(dependency)
            if match is None:
                continue
            name, extras, specifier = match.groups()
            if name not in names or specifier != expected_bounds:
                raise ValueError(f"Unexpected internal dependency in {path}: {dependency}")
            replacement = f"{name}{extras or ''}{new_bounds}"
            if f'"{dependency}"' not in updated:
                raise ValueError(f"Unsupported dependency formatting in {path}: {dependency}")
            updated = updated.replace(f'"{dependency}"', f'"{replacement}"')
        edits[path] = updated
    for package in RUNTIME_MODULES:
        path = ROOT / "packages" / f"neuro-co-{package}" / "src/neuro_co" / package / "__init__.py"
        edits[path] = replace_once(
            path.read_text(),
            rf'^__version__ = "{re.escape(current)}"$',
            f'__version__ = "{new_version}"',
            path,
        )
    citation = ROOT / "CITATION.cff"
    edits[citation] = replace_once(
        citation.read_text(),
        rf"^version: {re.escape(current)}$",
        f"version: {new_version}",
        citation,
    )
    return current, edits


def check_lock(version: str) -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    names = {f"neuro-co-{package}" for package in PACKAGES}
    locked = {package["name"]: package for package in lock["package"] if package["name"] in names}
    if set(locked) != names or any(package["version"] != version for package in locked.values()):
        raise ValueError("Workspace versions in uv.lock differ; run uv lock")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify metadata and locked versions")
    action.add_argument("version", nargs="?", help="set a stable X.Y.Z version and run uv lock")
    args = parser.parse_args()
    try:
        current, edits = plan(args.version)
        if args.check:
            check_lock(current)
            print(f"Version {current}: all 10 packages, runtime strings, citation and lock agree.")
            return 0
        if shutil.which("uv") is None:
            raise ValueError("Install uv before changing the version")
        originals = {path: path.read_bytes() for path in edits}
        lock = ROOT / "uv.lock"
        originals[lock] = lock.read_bytes()
        try:
            for path, content in edits.items():
                path.write_text(content)
            subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
            plan()
            check_lock(args.version)
        except BaseException:
            for path, content in originals.items():
                path.write_bytes(content)
            raise
        print(f"Version {current} -> {args.version}. Metadata and uv.lock updated.")
        return 0
    except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Version check failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
