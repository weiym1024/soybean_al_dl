#!/usr/bin/env python3

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SHA256SUMS.tsv"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


with MANIFEST.open() as f:
    records = list(csv.DictReader(f, delimiter="\t"))

failed = 0

for row in records:
    path = ROOT / row["path"]

    if not path.is_file():
        print("MISSING:", row["path"])
        failed += 1
        continue

    digest = sha256(path)

    if digest != row["sha256"]:
        print("MISMATCH:", row["path"])
        failed += 1

print(f"Checked {len(records)} files")
print("PASS" if failed == 0 else f"FAIL: {failed} files")

raise SystemExit(1 if failed else 0)
