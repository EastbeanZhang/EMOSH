import json

import lmdb
import numpy as np
import pytest
from PIL import Image

from prepare_dataset import prepare_record


def test_prepared_record_uses_relative_paths_and_exact_frame_order(tmp_path):
    source = tmp_path / "tracking"
    (source / "input").mkdir(parents=True)
    keys = ["frame_000002", "frame_000003"]
    np.save(source / "optim_tracking.npy", {key: {} for key in keys})
    np.save(source / "id_share_params.npy", {"shape": np.zeros(2)})
    for key in keys:
        Image.new("RGB", (64, 64), (10, 20, 30)).save(
            source / "input" / f"{key.removeprefix('frame_')}.png"
        )
    output = tmp_path / "dataset"
    info = prepare_record(source, output, "sample")
    assert info["frames_keys"] == keys
    assert info["video_dir"] == "sample"
    assert str(tmp_path) not in (output / "videos_info.json").read_text()
    assert json.loads((output / "videos_split.json").read_text())["train"] == ["sample"]
    with lmdb.open(str(output / "sample/image_lmdb"), readonly=True) as env:
        with env.begin() as txn:
            assert txn.get(b"frame_000002/full_crop_image") is not None
    with pytest.raises(FileExistsError):
        prepare_record(source, output, "sample")


def test_missing_frame_fails_before_writing_record(tmp_path):
    source = tmp_path / "tracking"
    source.mkdir()
    np.save(source / "optim_tracking.npy", {"000000": {}})
    np.save(source / "id_share_params.npy", {})
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_record(source, tmp_path / "dataset", "sample")

    assert not (tmp_path / "dataset/sample").exists()
