"""Dataset installer with dry-run planning and optional downloads."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from viewtrust.datasets.manifest import DatasetEntry, load_dataset_manifest
from viewtrust.utils.paths import ensure_child_path


@dataclass(frozen=True)
class DatasetInstallPlan:
    name: str
    url: str
    target_dir: Path
    archive_path: Path
    exists: bool
    sha256: str | None
    archive_type: str | None
    optional: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "url": self.url,
            "target_dir": str(self.target_dir),
            "archive_path": str(self.archive_path),
            "exists": self.exists,
            "sha256": self.sha256,
            "archive_type": self.archive_type,
            "optional": self.optional,
        }


def _archive_name(entry: DatasetEntry) -> str:
    filename = entry.url.rstrip("/").split("/")[-1]
    return filename or f"{entry.name}.archive"


def build_install_plan(manifest_path: Path, data_root: Path) -> list[DatasetInstallPlan]:
    manifest = load_dataset_manifest(manifest_path)
    data_root = data_root.resolve()
    downloads_dir = data_root / "_downloads"
    plans: list[DatasetInstallPlan] = []

    for entry in manifest.datasets:
        target_dir = (data_root / entry.target_subdir).resolve()
        archive_path = (downloads_dir / _archive_name(entry)).resolve()
        ensure_child_path(data_root, target_dir)
        ensure_child_path(data_root, archive_path)
        plans.append(
            DatasetInstallPlan(
                name=entry.name,
                url=entry.url,
                target_dir=target_dir,
                archive_path=archive_path,
                exists=target_dir.exists(),
                sha256=entry.sha256,
                archive_type=entry.archive_type,
                optional=entry.optional,
            )
        )

    return plans


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _validate_archive_member(target_dir: Path, member_name: str) -> None:
    member_path = (target_dir / member_name).resolve()
    ensure_child_path(target_dir, member_path)


def _extract(archive_path: Path, target_dir: Path, archive_type: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_type in {"tar", "tar.gz", "tgz"}:
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                _validate_archive_member(target_dir, member.name)
            archive.extractall(target_dir)
    elif archive_type == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                _validate_archive_member(target_dir, member)
            archive.extractall(target_dir)
    elif archive_type in {None, "none"}:
        shutil.copy2(archive_path, target_dir / archive_path.name)
    else:
        raise ValueError(f"unsupported archive_type: {archive_type}")


def install_from_manifest(
    manifest_path: Path,
    data_root: Path,
    *,
    download: bool,
    extract: bool,
) -> list[DatasetInstallPlan]:
    """Install datasets from a manifest, or return the dry-run plan."""

    plans = build_install_plan(manifest_path, data_root)
    if not download:
        return plans

    for plan in plans:
        if plan.exists:
            continue
        _download(plan.url, plan.archive_path)
        if plan.sha256 and _sha256(plan.archive_path) != plan.sha256:
            raise ValueError(f"sha256 mismatch for {plan.name}: {plan.archive_path}")
        if extract:
            _extract(plan.archive_path, plan.target_dir, plan.archive_type)

    return build_install_plan(manifest_path, data_root)
