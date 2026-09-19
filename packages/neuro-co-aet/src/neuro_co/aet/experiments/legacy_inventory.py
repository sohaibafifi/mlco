"""Create a checksummed, read-only inventory of legacy AET evidence.

The script hashes source files in place. It does not copy, edit, or normalize
them. The resulting JSONL manifest is a local forensic index, not proof that a
historical measurement is scientifically qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "aet-legacy-manifest/v1"
DEFAULT_OUTPUT = Path("experiments/aet-journal/legacy/legacy-manifest.jsonl")


@dataclass(frozen=True)
class Source:
    source_id: str
    path: Path
    trust: str
    note: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def git_value(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None


def git_snapshot(repo: Path) -> dict[str, Any]:
    status = git_value(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "path": str(repo.resolve()),
        "head": git_value(repo, "rev-parse", "HEAD"),
        "branch": git_value(repo, "branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": sha256_text(status or ""),
    }


def iter_source_files(source: Source) -> Iterable[tuple[Path, str]]:
    path = source.path
    if path.is_file() or path.is_symlink():
        yield path, path.name
        return
    if not path.is_dir():
        raise FileNotFoundError(f"legacy source does not exist: {path}")
    for candidate in sorted(path.rglob("*"), key=lambda p: str(p)):
        if candidate.is_file() or candidate.is_symlink():
            yield candidate, candidate.relative_to(path).as_posix()


def file_record(source: Source, path: Path, relative_path: str) -> dict[str, Any]:
    stat = path.lstat()
    if path.is_symlink():
        target = os.readlink(path)
        return {
            "record_type": "file",
            "source_id": source.source_id,
            "trust": source.trust,
            "source_root": str(source.path.resolve()),
            "relative_path": relative_path,
            "absolute_path": str(path.absolute()),
            "kind": "symlink",
            "symlink_target": target,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_text(target),
        }
    return {
        "record_type": "file",
        "source_id": source.source_id,
        "trust": source.trust,
        "source_root": str(source.path.resolve()),
        "relative_path": relative_path,
        "absolute_path": str(path.resolve()),
        "kind": "regular",
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def source_specs(legacy_root: Path, pdf: Path) -> list[Source]:
    return [
        Source(
            "submitted_pdf",
            pdf,
            "authoritative-submitted-document",
            "Exact four-page CIKM submission supplied by the author.",
        ),
        Source(
            "legacy_cvrp50_runs",
            legacy_root / "logs/aet/aet_aet_cvrp50",
            "unqualified-legacy-run-evidence",
            "Five-seed local run tree. Values do not reproduce the submission.",
        ),
        Source(
            "legacy_training_aggregate",
            legacy_root / "logs/aet/aet_train_cvrp50_aggregated.json",
            "unqualified-derived-summary",
            "Derived median summary, not raw cumulative training accounting.",
        ),
        Source(
            "legacy_test_instances",
            legacy_root / "data.rf/cvrp/test_50.npz",
            "legacy-dataset-requires-validation",
            "Historical CVRP50 test instances.",
        ),
        Source(
            "legacy_reference_solutions",
            legacy_root / "data.rf/cvrp/test_50_sol_pyvrp.npz",
            "legacy-reference-requires-independent-validation",
            "Historical PyVRP sidecar, not an independent certified reference.",
        ),
        Source(
            "untracked_aet_release",
            legacy_root / "aet_release",
            "untracked-snapshot-not-authoritative",
            "Untracked release tree that differs from the nearest committed branch.",
        ),
    ]


def write_manifest(
    output: Path,
    sources: list[Source],
    legacy_root: Path,
    *,
    force: bool,
) -> tuple[int, str]:
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    for source in sources:
        if not source.path.exists() and not source.path.is_symlink():
            raise FileNotFoundError(f"legacy source does not exist: {source.path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    count = 0
    header = {
        "record_type": "inventory",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "generator": str(Path(__file__).resolve()),
        "generator_git": git_snapshot(Path.cwd()),
        "legacy_repository_git": git_snapshot(legacy_root),
        "source_count": len(sources),
        "sources": [
            {
                **asdict(source),
                "path": str(source.path.resolve()),
            }
            for source in sources
        ],
        "qualification": (
            "A checksum establishes identity only. Every numerical result remains "
            "unqualified until units, hardware, backend, quality, and analysis are replayed."
        ),
    }
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for source in sources:
            for path, relative_path in iter_source_files(source):
                handle.write(
                    json.dumps(
                        file_record(source, path, relative_path),
                        sort_keys=True,
                    )
                    + "\n"
                )
                count += 1
    temp.replace(output)
    manifest_hash = sha256_file(output)
    checksum_path = output.parent / "MANIFEST.sha256"
    checksum_path.write_text(f"{manifest_hash}  {output.name}\n", encoding="utf-8")
    return count, manifest_hash


def verify_manifest(manifest: Path) -> dict[str, Any]:
    """Verify the manifest checksum and every referenced legacy file."""

    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read legacy manifest {manifest}: {exc}") from exc
    if not lines:
        raise ValueError("legacy manifest is empty")
    try:
        records = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError(f"legacy manifest contains invalid JSON: {exc}") from exc
    header = records[0]
    if (
        not isinstance(header, dict)
        or header.get("record_type") != "inventory"
        or header.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("legacy manifest header or schema is invalid")
    file_records = records[1:]
    if not file_records:
        raise ValueError("legacy manifest has no file records")
    sources = header.get("sources")
    if not isinstance(sources, list) or len(sources) != header.get("source_count"):
        raise ValueError("legacy manifest source inventory is inconsistent")
    expected_source_ids = {
        source.get("source_id") for source in sources if isinstance(source, dict)
    }
    if len(expected_source_ids) != len(sources) or None in expected_source_ids:
        raise ValueError("legacy manifest source identifiers are invalid")

    expected_checksum_path = manifest.parent / "MANIFEST.sha256"
    try:
        checksum_fields = expected_checksum_path.read_text(encoding="utf-8").split()
    except OSError as exc:
        raise ValueError(f"cannot read {expected_checksum_path}: {exc}") from exc
    if len(checksum_fields) != 2 or checksum_fields[1] != manifest.name:
        raise ValueError("MANIFEST.sha256 has an invalid format or filename")
    actual_manifest_sha = sha256_file(manifest)
    if checksum_fields[0].lower() != actual_manifest_sha:
        raise ValueError("legacy manifest checksum mismatch")

    seen: set[tuple[str, str]] = set()
    observed_source_ids: set[str] = set()
    for index, record in enumerate(file_records, start=2):
        if not isinstance(record, dict) or record.get("record_type") != "file":
            raise ValueError(f"legacy manifest line {index} is not a file record")
        key = (str(record.get("source_id")), str(record.get("relative_path")))
        if key in seen:
            raise ValueError(f"duplicate legacy file record at line {index}: {key}")
        seen.add(key)
        observed_source_ids.add(key[0])
        path = Path(str(record.get("absolute_path", "")))
        kind = record.get("kind")
        if kind == "regular":
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"legacy regular file is missing or changed kind: {path}")
            if path.stat().st_size != record.get("size_bytes"):
                raise ValueError(f"legacy file size changed: {path}")
            actual_sha = sha256_file(path)
        elif kind == "symlink":
            if not path.is_symlink():
                raise ValueError(f"legacy symlink is missing or changed kind: {path}")
            actual_sha = sha256_text(os.readlink(path))
        else:
            raise ValueError(f"unknown legacy file kind at line {index}: {kind!r}")
        if actual_sha != record.get("sha256"):
            raise ValueError(f"legacy file checksum changed: {path}")
    if observed_source_ids != expected_source_ids:
        raise ValueError("legacy manifest file records do not cover every source")
    return {
        "manifest": str(manifest),
        "manifest_sha256": actual_manifest_sha,
        "file_records": len(file_records),
        "verified": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, help="legacy experiment repository")
    parser.add_argument("--pdf", type=Path, help="submitted paper to include in the inventory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the existing output manifest and all referenced files",
    )
    args = parser.parse_args(argv)
    if not args.verify and (args.legacy_root is None or args.pdf is None):
        parser.error("creating an inventory requires --legacy-root and --pdf")
    return args


def main() -> int:
    args = parse_args()
    if args.verify:
        print(json.dumps(verify_manifest(args.output.resolve()), sort_keys=True))
        return 0
    sources = source_specs(args.legacy_root.resolve(), args.pdf.resolve())
    count, manifest_hash = write_manifest(
        args.output,
        sources,
        args.legacy_root.resolve(),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_records": count,
                "sha256": manifest_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
