#!/usr/bin/env python3
"""Verify that every offline dataset/model artifact is present and intact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also calculate all SHA-256 hashes")
    parser.add_argument("--assemble-model", action="store_true", help="Reassemble and verify the local sentiment model")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest_path = root / "data" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.exists():
            failures.append(f"MISSING {item['path']}")
            continue
        if path.stat().st_size != item["size"]:
            failures.append(f"SIZE {item['path']}: {path.stat().st_size} != {item['size']}")
            continue
        if args.full and digest(path) != item["sha256"]:
            failures.append(f"HASH {item['path']}")
    if args.assemble_model:
        try:
            from crypto_rl.news import assemble_offline_model
            assemble_offline_model(root / "models" / "financial_sentiment")
        except Exception as exc:
            failures.append(f"MODEL {type(exc).__name__}: {exc}")
    if failures:
        print("Offline asset verification FAILED:")
        print("\n".join(f"  - {item}" for item in failures))
        return 1
    mode = "size + SHA-256" if args.full else "presence + size"
    print(f"Verified {len(manifest['files'])} files ({mode}).")
    print(json.dumps(manifest.get("coverage", {}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
