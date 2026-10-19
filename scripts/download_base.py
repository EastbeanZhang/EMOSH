import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Download the public Wan initialization files.")
    parser.add_argument("--output", type=Path, default=ROOT / "models/wan-base")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "assets/base_manifest.json").read_text())
    if not args.verify_only:
        snapshot_download(
            repo_id=manifest["repository"],
            revision=manifest["revision"],
            allow_patterns=[item["path"] for item in manifest["files"]],
            local_dir=args.output,
            token=False,
            max_workers=4,
        )
    for item in manifest["files"]:
        path = args.output / item["path"]
        if path.stat().st_size != item["size"]:
            raise ValueError(f"Unexpected size: {item['path']}")
        if "sha256" in item:
            digest = hashlib.sha256()
        else:
            digest = hashlib.sha1(f"blob {item['size']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != item.get("sha256", item.get("git_blob_sha1")):
            raise ValueError(f"Checksum mismatch: {item['path']}")
        print(f"Verified {item['path']}")


if __name__ == "__main__":
    main()
