"""Privacy-aware, checksummed provenance for AET experiment runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "aet-provenance/v1"
_RUN_STATUSES = {"prepared", "running", "completed", "failed", "interrupted"}


@dataclass(frozen=True)
class ArtifactFingerprint:
    """Identity of an input or output used by a run."""

    path: str
    size_bytes: int
    sha256: str


@dataclass
class RunMetadata:
    """Serializable provenance record without public host identifiers."""

    schema_version: str
    run_id: str
    host_id: str
    timestamp_start: str
    timestamp_end: str
    status: str
    failure_reason: str | None
    git_sha: str | None
    git_dirty: bool
    dirty_patch_sha256: str | None
    python_version: str
    python_executable: str
    platform: str
    execution_layer: str
    cpu: str
    memory_total_b: int | None
    gpus: list[dict[str, Any]]
    cuda_version: str | None
    environment_hash: str
    library_versions: dict[str, str] = field(default_factory=dict)
    lockfiles: dict[str, ArtifactFingerprint] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    config_sha256: str = ""
    artifacts: dict[str, ArtifactFingerprint] = field(default_factory=dict)
    runtime_controls: dict[str, Any] = field(default_factory=dict)
    measurement: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    hostname: str | None = None

    @property
    def env_hash(self) -> str:
        """Compatibility alias for the old field name."""

        return self.environment_hash


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _aware_timestamp(value: str, *, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")
    return parsed


def _git_output(root: Path, *args: str, binary: bool = False) -> bytes | str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args],
        stderr=subprocess.DEVNULL,
        text=not binary,
    )


def _git_snapshot(excluded: Iterable[Path] = ()) -> tuple[str | None, bool, str | None]:
    """Return commit, dirty flag, and a content hash of tracked plus untracked edits."""

    try:
        root_raw = _git_output(Path.cwd(), "rev-parse", "--show-toplevel")
        root = Path(str(root_raw).strip()).resolve()
        sha = str(_git_output(root, "rev-parse", "HEAD")).strip()
        diff = _git_output(root, "diff", "--binary", "HEAD", binary=True)
        untracked = _git_output(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            binary=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None, False, None

    excluded_resolved = {path.resolve() for path in excluded}
    digest = hashlib.sha256()
    diff_bytes = diff if isinstance(diff, bytes) else diff.encode("utf-8")
    digest.update(diff_bytes)
    raw_untracked = untracked if isinstance(untracked, bytes) else untracked.encode("utf-8")
    included_untracked = False
    for raw in sorted(part for part in raw_untracked.split(b"\0") if part):
        relative = raw.decode("utf-8", errors="surrogateescape")
        path = (root / relative).resolve()
        if path in excluded_resolved or not path.is_file():
            continue
        included_untracked = True
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(bytes.fromhex(_sha256_file(path)))
    dirty = bool(diff_bytes) or included_untracked
    if not dirty:
        return sha, False, None
    return sha, True, digest.hexdigest()


def _gpu_inventory() -> tuple[list[dict[str, Any]], str | None]:
    """Return non-identifying GPU facts and a hashed physical device ID."""

    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
    except Exception:
        return [], None
    gpus: list[dict[str, Any]] = []
    cuda: str | None = None
    try:
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
            device_id_hash: str | None = None
            try:
                raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(raw_uuid, bytes):
                    raw_uuid = raw_uuid.decode("utf-8", errors="replace")
                device_id_hash = _sha256_bytes(str(raw_uuid).encode("utf-8"))
            except Exception:
                pass
            power_limit_w: float | None = None
            with suppress(Exception):
                power_limit_w = float(pynvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0
            gpus.append(
                {
                    "index": index,
                    "name": str(name),
                    "memory_total_b": memory,
                    "device_id_sha256": device_id_hash,
                    "power_limit_w": power_limit_w,
                }
            )
        cuda_raw = int(pynvml.nvmlSystemGetCudaDriverVersion())
        cuda = f"{cuda_raw // 1000}.{(cuda_raw % 1000) // 10}"
    finally:
        with suppress(Exception):
            pynvml.nvmlShutdown()
    return gpus, cuda


def _execution_layer() -> str:
    if sys.platform == "win32":
        return "windows-native"
    if sys.platform.startswith("linux"):
        try:
            release = Path("/proc/version").read_text().lower()
        except OSError:
            release = ""
        return "wsl2" if "microsoft" in release else "native-linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform


def _memory_total_b() -> int | None:
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return int(value.strip())
        except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError):
            return None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def _library_versions(extra_libs: list[str] | None = None) -> dict[str, str]:
    names = ["neuro-co-aet", "neuro-co-core", "torch", "numpy", "codecarbon"]
    if extra_libs:
        names.extend(extra_libs)
    versions: dict[str, str] = {}
    for name in dict.fromkeys(names):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _environment_hash(libraries: dict[str, str]) -> str:
    """Hash reproducibility facts only, never environment variable values."""

    return _json_sha256(
        {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
            "platform": platform.platform(),
            "libraries": libraries,
        }
    )


def _public_artifact_path(path: Path) -> str:
    """Return a portable label without exposing host-specific parent paths."""

    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _fingerprint(path: Path) -> ArtifactFingerprint:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {path}")
    return ArtifactFingerprint(
        path=_public_artifact_path(path),
        size_bytes=resolved.stat().st_size,
        sha256=_sha256_file(resolved),
    )


def _fingerprints(
    paths: dict[str, str | os.PathLike[str]] | None,
) -> dict[str, ArtifactFingerprint]:
    return {name: _fingerprint(Path(path)) for name, path in (paths or {}).items()}


def record_run(
    output_dir: str | os.PathLike[str] | None = None,
    *,
    run_id: str = "unspecified",
    host_id: str = "unspecified",
    started_at: str | None = None,
    ended_at: str | None = None,
    status: str = "prepared",
    failure_reason: str | None = None,
    seeds: dict[str, int] | None = None,
    config: dict[str, Any] | None = None,
    artifacts: dict[str, str | os.PathLike[str]] | None = None,
    lockfiles: dict[str, str | os.PathLike[str]] | None = None,
    runtime_controls: dict[str, Any] | None = None,
    measurement: dict[str, Any] | None = None,
    extra_libs: list[str] | None = None,
    extra: dict[str, Any] | None = None,
    allow_dirty: bool = False,
    include_hostname: bool = False,
    overwrite: bool = False,
) -> RunMetadata:
    """Capture provenance and optionally write an immutable `provenance.json`.

    Clean Git state is required by default. Development callers may explicitly
    pass `allow_dirty=True`; the content fingerprint is then recorded.
    """

    now = datetime.now(UTC).isoformat()
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(host_id, str) or not host_id.strip():
        raise ValueError("host_id must be a non-empty string")
    if status not in _RUN_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(_RUN_STATUSES))}")
    if status in {"failed", "interrupted"} and not failure_reason:
        raise ValueError(f"status {status!r} requires failure_reason")
    if status not in {"failed", "interrupted"} and failure_reason is not None:
        raise ValueError(f"status {status!r} cannot carry failure_reason")
    if not isinstance(allow_dirty, bool):
        raise TypeError("allow_dirty must be a boolean")
    if not isinstance(include_hostname, bool):
        raise TypeError("include_hostname must be a boolean")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    start_value = started_at or now
    end_value = ended_at or now
    start_timestamp = _aware_timestamp(start_value, field_name="started_at")
    end_timestamp = _aware_timestamp(end_value, field_name="ended_at")
    if end_timestamp < start_timestamp:
        raise ValueError("ended_at must not precede started_at")
    output_path = Path(output_dir).resolve() if output_dir is not None else None
    excluded = [output_path / "provenance.json"] if output_path is not None else []
    sha, dirty, patch_hash = _git_snapshot(excluded)
    if not allow_dirty and sha is None:
        raise RuntimeError("measured run requires an available Git repository")
    if dirty and not allow_dirty:
        raise RuntimeError("measured run requires a clean Git worktree")
    libraries = _library_versions(extra_libs)
    artifact_records = _fingerprints(artifacts)
    lockfile_records = _fingerprints(lockfiles)
    gpus, cuda = _gpu_inventory()
    config_value = dict(config or {})
    metadata = RunMetadata(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        host_id=host_id,
        timestamp_start=start_value,
        timestamp_end=end_value,
        status=status,
        failure_reason=failure_reason,
        git_sha=sha,
        git_dirty=dirty,
        dirty_patch_sha256=patch_hash,
        python_version=platform.python_version(),
        python_executable=Path(sys.executable).name,
        platform=platform.platform(),
        execution_layer=_execution_layer(),
        cpu=platform.processor() or platform.machine(),
        memory_total_b=_memory_total_b(),
        gpus=gpus,
        cuda_version=cuda,
        environment_hash=_environment_hash(libraries),
        library_versions=libraries,
        lockfiles=lockfile_records,
        seeds={key: int(value) for key, value in (seeds or {}).items()},
        config=config_value,
        config_sha256=_json_sha256(config_value),
        artifacts=artifact_records,
        runtime_controls=dict(runtime_controls or {}),
        measurement=dict(measurement or {}),
        extra=dict(extra or {}),
        hostname=platform.node() if include_hostname else None,
    )
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / "provenance.json"
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite provenance record: {target}")
        target.write_text(json.dumps(asdict(metadata), indent=2, default=str) + "\n")
    return metadata
