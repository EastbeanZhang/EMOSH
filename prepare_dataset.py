import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
import shutil
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image

from models.utils.dataset_index import DatasetIndex, IndexWriter, export_json, file_lock

DEFAULT_PROMPT = "视频中的人在做动作"


def _image_size(frames):
    size = None
    for frame in frames:
        with Image.open(frame) as image:
            image.load()
            if image.mode != "RGB" or (size is not None and image.size != size):
                raise ValueError("All input PNGs must be RGB and have the same dimensions.")
            size = image.size
    return size


def _source(tracking, split, prompt, keys=None):
    if keys is None:
        parameters = np.load(tracking / "optim_tracking.npy", allow_pickle=True).item()
        if not isinstance(parameters, dict):
            raise ValueError("Tracking parameters must be a frame dictionary")
        identity = np.load(tracking / "id_share_params.npy", allow_pickle=True).item()
        if not isinstance(identity, dict):
            raise ValueError("Identity parameters must be a dictionary")
        keys = list(parameters)
    frames = []
    for key in keys:
        path = tracking / "input" / f"{key}.png"
        if not path.is_file():
            path = tracking / "input" / f"{str(key).removeprefix('frame_')}.png"
        frames.append(path)
    if not keys or not all(frame.is_file() for frame in frames):
        raise ValueError("Every tracked frame needs an input PNG; run Tracker with --save-images.")
    identity = tracking / "id_share_params.npy"
    if not identity.is_file():
        raise FileNotFoundError("Tracker identity parameters are missing")
    metadata_path = tracking / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata and not metadata.get("completed", False):
        raise ValueError("Tracker has not completed this record.")
    if metadata.get("frame_count", len(keys)) != len(keys):
        raise ValueError("Tracker frame count does not match the parameters")
    if "frames_keys" in metadata and metadata["frames_keys"] != keys:
        raise ValueError("Tracker frame order does not match the parameters")
    if "source_frame_indices" in metadata and len(metadata["source_frame_indices"]) != len(keys):
        raise ValueError("Source frame indices do not match the parameters")
    fps = metadata.get("fps")
    if fps is not None and (
        not isinstance(fps, (int, float))
        or isinstance(fps, bool)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("FPS must be a positive finite number")
    signature = hashlib.sha256(json.dumps([1, split, prompt]).encode())
    files = [tracking / "optim_tracking.npy", identity, *frames]
    if metadata_path.is_file():
        files.append(metadata_path)
    for path in files:
        stat = path.stat()
        signature.update(json.dumps([path.name, stat.st_size, stat.st_mtime_ns]).encode())
    return keys, frames, metadata, signature.hexdigest()


def _pack(job):
    tracking, output = Path(job["tracking"]), Path(job["output"])
    record_id, split, prompt = job["id"], job["split"], job["prompt"]
    if not isinstance(record_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
        raise ValueError("Record IDs may contain letters, numbers, underscores and hyphens.")
    if split not in ("train", "valid") or not isinstance(prompt, str):
        raise ValueError("Provide a train/valid split and a text prompt.")
    bucket = hashlib.sha256(record_id.encode()).hexdigest()[:2]
    relative = job.get("directory") or f"records/{bucket}/{record_id}"
    destination = output / relative
    if not destination.resolve().is_relative_to(output.resolve()):
        raise ValueError("Record directory must be inside the dataset root")
    size = None
    receipt = destination / "record.json"
    prior_keys = None
    if job.get("resume") and receipt.is_file():
        prior_keys = json.loads(receipt.read_text())["info"]["frames_keys"]
    keys, frames, metadata, signature = _source(tracking, split, prompt, prior_keys)
    if destination.exists() and not job.get("resume"):
        raise FileExistsError(f"Record already exists: {record_id}; use --resume to verify it.")
    if not destination.exists():
        size = _image_size(frames)
    with file_lock(output / ".locks" / f"{bucket}.lock"):
        if destination.exists():
            receipt = destination / "record.json"
            if not job.get("resume") or not receipt.is_file():
                raise FileExistsError(f"Existing record has no resumable receipt: {record_id}")
            record = json.loads(receipt.read_text())
            if record["id"] != record_id or record["signature"] != signature:
                raise ValueError(f"Source or packing settings changed for record: {record_id}")
            for name, expected in record["file_sizes"].items():
                path = destination / name
                if not path.is_file() or path.stat().st_size != expected:
                    raise ValueError(f"Packed record is damaged: {record_id}/{name}")
            return {**record, "status": "skipped"}
        if size is None:
            size = _image_size(frames)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if job.get("resume"):
            for partial in destination.parent.glob(f".{record_id}.partial-*"):
                if partial.is_dir() and not partial.is_symlink():
                    shutil.rmtree(partial)
        temporary = Path(tempfile.mkdtemp(prefix=f".{record_id}.partial-", dir=destination.parent))
        try:
            capacity = max(
                64 * 1024**2,
                sum(frame.stat().st_size for frame in frames) * 2 + len(frames) * 16384,
            )
            environment = lmdb.open(str(temporary / "image_lmdb"), map_size=capacity)
            try:
                with environment.begin(write=True) as transaction:
                    for key, frame in zip(keys, frames):
                        transaction.put(f"{key}/full_crop_image".encode(), frame.read_bytes())
            finally:
                environment.close()
            for name in ("optim_tracking.npy", "id_share_params.npy"):
                shutil.copyfile(tracking / name, temporary / name)
            (temporary / "prompt.txt").write_text(prompt + "\n")
            if _source(tracking, split, prompt, keys)[3] != signature:
                raise ValueError("Source changed while packing")
            info = {
                "video_dir": relative,
                "frames_keys": keys,
                "frames_number": len(keys),
                "width": size[0],
                "height": size[1],
            }
            for name in ("fps", "source_frame_indices"):
                if name in metadata:
                    info[name] = metadata[name]
            record = {
                "id": record_id,
                "split": split,
                "info": info,
                "signature": signature,
                "file_sizes": {
                    name: (temporary / name).stat().st_size
                    for name in (
                        "image_lmdb/data.mdb",
                        "optim_tracking.npy",
                        "id_share_params.npy",
                        "prompt.txt",
                    )
                },
            }
            (temporary / "record.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
            os.rename(temporary, destination)
            return {**record, "status": "written"}
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def prepare_record(tracking, output, record_id, split="train", prompt=DEFAULT_PROMPT, resume=False):
    directory = None
    if (Path(output) / "dataset.sqlite3").exists() or (Path(output) / "videos_info.json").exists():
        index = DatasetIndex(output)
        try:
            if record_id in index:
                directory = index[record_id].get("video_dir", ".")
        finally:
            index.close()
    result = _pack(
        {
            "tracking": str(tracking),
            "output": str(output),
            "id": record_id,
            "split": split,
            "prompt": prompt,
            "resume": resume,
            "directory": directory,
        }
    )
    writer = IndexWriter(output)
    try:
        writer.add([result])
    finally:
        writer.close()
    return result["info"]


def prepare_manifest(
    manifest, output, workers=4, resume=False, split="train", prompt=DEFAULT_PROMPT
):
    manifest, output = Path(manifest), Path(output)
    if workers < 1:
        raise ValueError("--workers must be positive")
    writer = IndexWriter(output)
    counts = {"written": 0, "skipped": 0, "failed": 0}
    error_path = output / f"packing-errors-{uuid.uuid4().hex[:12]}.jsonl"
    seen = set()
    buffered = []
    last_commit = time.monotonic()

    def failure(line, record_id, error, message=None):
        counts["failed"] += 1
        with error_path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "line": line,
                        "id": record_id,
                        "error_type": type(error).__name__,
                        "error": str(error) if message is None else message,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def jobs():
        with manifest.open() as stream:
            for line, text in enumerate(stream, 1):
                if not text.strip():
                    continue
                record_id = None
                try:
                    value = json.loads(text)
                    record_id = value["id"]
                    if not isinstance(record_id, str) or record_id in seen:
                        raise ValueError("Manifest IDs must be unique strings")
                    seen.add(record_id)
                    tracking = Path(value["tracking"])
                    if not tracking.is_absolute():
                        tracking = manifest.parent / tracking
                    yield (
                        line,
                        {
                            "id": record_id,
                            "tracking": str(tracking),
                            "output": str(output),
                            "split": value.get("split", split),
                            "prompt": value.get("prompt", prompt),
                            "resume": resume,
                            "directory": writer.directory(record_id),
                        },
                    )
                except (KeyError, TypeError, ValueError) as error:
                    failure(line, record_id, error)

    try:
        iterator = iter(jobs())
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            pending = {}
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < 2 * workers:
                    try:
                        line, job = next(iterator)
                        pending[executor.submit(_pack, job)] = (line, job)
                    except StopIteration:
                        exhausted = True
                if not pending:
                    break
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in completed:
                    line, job = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        message = str(error).replace(
                            str(Path(job["tracking"]).resolve()), "[tracking]"
                        )
                        message = message.replace(str(output.resolve()), "[output]")
                        failure(line, job["id"], error, message)
                    else:
                        buffered.append(record)
                        counts[record["status"]] += 1
                if buffered and (len(buffered) >= 128 or time.monotonic() - last_commit >= 1):
                    writer.add(buffered)
                    buffered.clear()
                    last_commit = time.monotonic()
                    print(json.dumps(counts), flush=True)
            if buffered:
                writer.add(buffered)
    finally:
        writer.close()
    if counts["failed"]:
        print(f"Failed records: {error_path}", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Pack tracked clips into an EMOSH dataset.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tracking")
    mode.add_argument("--manifest", help="JSONL with id, tracking and optional split/prompt")
    mode.add_argument("--index-only", action="store_true", help="Index an existing JSON dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id")
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--export-json", action="store_true", help="Export legacy JSON indexes after packing"
    )
    args = parser.parse_args()
    if args.tracking and not args.id:
        parser.error("--id is required with --tracking")
    if not args.tracking and args.id:
        parser.error("--id is only used with --tracking")
    failed = False
    if args.manifest:
        result = prepare_manifest(
            args.manifest, args.output, args.workers, args.resume, args.split, args.prompt
        )
        failed = bool(result["failed"])
    elif args.index_only:
        if (
            not (Path(args.output) / "dataset.sqlite3").is_file()
            and not (Path(args.output) / "videos_info.json").is_file()
        ):
            parser.error("No existing dataset index found under --output")
        writer = IndexWriter(args.output)
        writer.close()
        index = DatasetIndex(args.output)
        try:
            result = {"records": len(index)}
        finally:
            index.close()
    else:
        result = prepare_record(
            args.tracking, args.output, args.id, args.split, args.prompt, args.resume
        )
    if args.export_json:
        export_json(args.output)
    print(json.dumps(result, ensure_ascii=False))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
