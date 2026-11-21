"""Verify originals against the immutable Phase 2 snapshot without changing files."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "preservation" / "original-files.json"
# Only these originals may be absent in a publication checkout. An existing
# excluded file must still match its original checksum.
PUBLICATION_EXCLUSIONS = {
    "Final_Report.pdf",
    "REVISED_TASK A AND B.pdf",
    "C-GRID.jpg",
    "Project_TWO.",
}


def original_path(relative):
    """Preserve Windows trailing dots instead of silently normalising names."""
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or PurePosixPath(relative).is_absolute()
        or ".." in parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"Unsafe manifest path: {relative!r}")
    root = str(ROOT)
    if os.name == "nt":
        root = "\\\\?\\" + root
    return os.path.join(root, *parts)


def verify(allow_excluded_missing=False):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = manifest["files"]
    if manifest["file_count"] != len(records):
        raise ValueError("Manifest file count is inconsistent")
    if len({record["path"] for record in records}) != len(records):
        raise ValueError("Manifest contains duplicate paths")
    repair_file = ROOT / "preservation" / "repairs.json"
    repair_records = json.loads(repair_file.read_text(encoding="utf-8"))["repairs"] if repair_file.exists() else []
    repairs = {record["path"]: record for record in repair_records}
    if len(repairs) != len(repair_records) or not set(repairs).issubset({r["path"] for r in records}):
        raise ValueError("Invalid repair inventory")
    verified, repaired, omitted, failures = [], [], [], []
    for record in records:
        relative = record["path"]
        path = original_path(relative)
        repair = repairs.get(relative)
        if repair:
            if repair["original_sha256"] != record["sha256"]:
                failures.append(f"REPAIR BASELINE MISMATCH: {relative}")
                continue
            backup = original_path(repair["original_backup"])
            if not os.path.isfile(backup):
                failures.append(f"ORIGINAL BACKUP MISSING: {relative}")
                continue
            with open(backup, "rb") as stream:
                backup_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if backup_hash != record["sha256"] or os.path.getsize(backup) != record["bytes"]:
                failures.append(f"ORIGINAL BACKUP CHANGED: {relative}")
                continue
        if not os.path.isfile(path):
            if allow_excluded_missing and relative in PUBLICATION_EXCLUSIONS:
                omitted.append(relative)
            else:
                failures.append(f"MISSING: {relative}")
            continue
        with open(path, "rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        expected_hash = repair["repaired_sha256"] if repair else record["sha256"]
        expected_bytes = repair["repaired_bytes"] if repair else record["bytes"]
        if os.path.getsize(path) != expected_bytes or digest != expected_hash:
            failures.append(f"CHANGED: {relative}")
        elif repair:
            repaired.append(relative)
        else:
            verified.append(relative)
    return verified, repaired, omitted, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-excluded-missing", action="store_true",
        help="For publication checkouts only: allow the four deliberately excluded originals to be absent",
    )
    args = parser.parse_args()
    verified, repaired, omitted, failures = verify(args.allow_excluded_missing)
    print(f"Verified {len(verified)} unchanged originals; {len(repaired)} recorded repairs with original backups; allowed missing {len(omitted)}; failures {len(failures)}")
    for message in failures:
        print(message)
    for relative in omitted:
        print(f"EXCLUDED FROM PUBLICATION: {relative}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
