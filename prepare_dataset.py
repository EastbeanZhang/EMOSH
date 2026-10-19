import argparse
import json
import re
import shutil
from pathlib import Path

import lmdb
import numpy as np
from PIL import Image


def prepare_record(tracking, output, record_id, split="train", prompt="视频中的人在做动作"):
    tracking, output = Path(tracking), Path(output)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
        raise ValueError("Record IDs may contain letters, numbers, underscores and hyphens.")
    parameters = np.load(tracking / "optim_tracking.npy", allow_pickle=True).item()
    keys = list(parameters)
    frames = []
    for key in keys:
        path = tracking / "input" / f"{key}.png"
        if not path.is_file():
            path = tracking / "input" / f"{str(key).removeprefix('frame_')}.png"
        frames.append(path)
    if not keys or not all(frame.is_file() for frame in frames):
        raise ValueError(
            "Run Tracker with --visualize --save-images; every tracked frame needs an input PNG."
        )
    if not (tracking / "id_share_params.npy").is_file():
        raise FileNotFoundError("Tracker identity parameters are missing")
    info_path, split_path = output / "videos_info.json", output / "videos_split.json"
    information = json.loads(info_path.read_text()) if info_path.exists() else {}
    splits = (
        json.loads(split_path.read_text()) if split_path.exists() else {"train": [], "valid": []}
    )
    destination = output / record_id
    if destination.exists():
        raise FileExistsError(destination)
    if record_id in information:
        raise ValueError("Record ID already exists in dataset metadata.")
    metadata_path = tracking / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if metadata and not metadata.get("completed", False):
        raise ValueError("Tracker has not completed this record.")
    size = None
    for frame in frames:
        with Image.open(frame) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError("Tracker input PNGs must be RGB.")
            if size is not None and image.size != size:
                raise ValueError("Every frame in a record must use the same fitting canvas.")
            size = image.size
    destination.mkdir(parents=True)
    environment = lmdb.open(str(destination / "image_lmdb"), map_size=2**40)
    try:
        with environment.begin(write=True) as transaction:
            for key, frame in zip(keys, frames):
                transaction.put(f"{key}/full_crop_image".encode(), frame.read_bytes())
    finally:
        environment.close()
    for name in ["optim_tracking.npy", "id_share_params.npy"]:
        shutil.copyfile(tracking / name, destination / name)
    (destination / "prompt.txt").write_text(prompt + "\n")
    information[record_id] = {
        "video_dir": record_id,
        "frames_keys": keys,
        "frames_number": len(keys),
        "width": size[0],
        "height": size[1],
    }
    for name in ["fps", "source_frame_indices"]:
        if name in metadata:
            information[record_id][name] = metadata[name]
    splits.setdefault(split, []).append(record_id)
    info_path.write_text(json.dumps(information, indent=2) + "\n")
    split_path.write_text(json.dumps(splits, indent=2) + "\n")
    return information[record_id]


def main():
    parser = argparse.ArgumentParser(
        description="Pack EMOSH-Tracker output into a training record."
    )
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--split", choices=["train", "valid"], default="train")
    parser.add_argument("--prompt", default="视频中的人在做动作")
    args = parser.parse_args()
    result = prepare_record(args.tracking, args.output, args.id, args.split, args.prompt)
    print(f"Prepared {result['frames_number']} frames at {result['width']}x{result['height']}.")


if __name__ == "__main__":
    main()
