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
    assert (output / "sample/prompt.txt").read_text().strip() == "视频中的人在做动作"
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


def test_converter_keeps_timing_without_the_local_input_path(tmp_path):
    source = tmp_path / "tracking"
    (source / "input").mkdir(parents=True)
    np.save(source / "optim_tracking.npy", {"frame_000002": {}})
    np.save(source / "id_share_params.npy", {})
    Image.new("RGB", (79, 65)).save(source / "input/000002.png")
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "completed": True,
                "fps": 25,
                "source_frame_indices": [2],
                "input": "local-video.mp4",
            }
        )
    )
    output = tmp_path / "dataset"
    info = prepare_record(source, output, "sample")
    assert info["fps"] == 25 and info["source_frame_indices"] == [2]
    assert (info["width"], info["height"]) == (79, 65)
    assert (
        "input" not in info and "local-video.mp4" not in (output / "videos_info.json").read_text()
    )


@pytest.mark.parametrize("invalid", ["size", "mode", "incomplete"])
def test_invalid_tracker_record_fails_without_creating_partial_output(tmp_path, invalid):
    source = tmp_path / "tracking"
    (source / "input").mkdir(parents=True)
    np.save(source / "optim_tracking.npy", {"frame_000000": {}, "frame_000001": {}})
    np.save(source / "id_share_params.npy", {})
    Image.new("RGB", (64, 64)).save(source / "input/000000.png")
    Image.new(
        "L" if invalid == "mode" else "RGB", (80, 64) if invalid == "size" else (64, 64)
    ).save(source / "input/000001.png")
    if invalid == "incomplete":
        (source / "metadata.json").write_text(json.dumps({"completed": False}))
    output = tmp_path / "dataset"
    with pytest.raises(ValueError):
        prepare_record(source, output, "sample")
    assert not output.exists()


def make_tracked_video(path, size):
    width, height = size
    (path / "input").mkdir(parents=True)
    parameters = {}
    intrinsic = np.array(
        [[50.0, 0, width / 2], [0, 50.0, height / 2], [0, 0, 1]], dtype=np.float32
    )[None]
    for i in range(9):
        key = f"frame_{i:06d}"
        parameters[key] = {
            "smplx_params": {
                "focal_px": np.array([[50.0]], dtype=np.float32),
                "focal_py": np.array([[50.0]], dtype=np.float32),
                "K_intrinsic": intrinsic.copy(),
                "w2c_RT": np.eye(4, dtype=np.float32)[None],
            },
            "full_crop": {
                "M_o2c-full": np.eye(3, dtype=np.float32)[None],
                "M_c2o-full": np.eye(3, dtype=np.float32)[None],
            },
        }
        for name in ["head_crop", "left_hand_crop", "right_hand_crop"]:
            parameters[key][name] = {"M_c2o": np.eye(3, dtype=np.float32)[None]}
        Image.new("RGB", size, (80, 120, 160)).save(path / f"input/{i:06d}.png")
    np.save(path / "optim_tracking.npy", parameters)
    np.save(path / "id_share_params.npy", {"shape": np.zeros((1, 1), dtype=np.float32)})


def test_native_dataset_loading_preserves_odd_sized_pixels(tmp_path):
    import torch

    from models.utils.data_utils import Tracked_VideoDataset

    tracking = tmp_path / "tracking"
    make_tracked_video(tracking, (65, 79))
    prepare_record(tracking, tmp_path / "data", "sample")
    dataset = Tracked_VideoDataset(base_path=str(tmp_path / "data"), split="train", num_frames=9)
    sample = dataset[0]
    assert len(sample["target_video"]) == 9 and sample["target_video"][0].shape == (80, 80, 3)
    pixel = sample["target_video"][0][30, 30]
    torch.testing.assert_close(pixel, torch.tensor([80, 120, 160]) / 255)
    intrinsic = sample["target_infos"][0]["smplx_params"]["K_intrinsic"][0]
    torch.testing.assert_close(intrinsic[:2, 2], torch.tensor([39.5, 39.5]))
    dataset.close_all_engines()


@pytest.mark.parametrize("driving_size", [(96, 64), (80, 80)])
def test_different_driving_size_stays_inside_the_reference_export_window(tmp_path, driving_size):
    import torch

    from models.utils.data_utils import Tracked_VideoDataset_infer

    for name, size in [("reference", (65, 79)), ("driving", driving_size)]:
        tracking = tmp_path / name
        make_tracked_video(tracking, size)
        prepare_record(tracking, tmp_path / "data", name)
    dataset = Tracked_VideoDataset_infer(
        base_path=str(tmp_path / "data"), split="test", num_frames=9
    )
    source = dataset._load_source_info("reference")
    target = dataset._load_target_info("driving")
    assert source["source_image"].shape == target["target_video"][0].shape == (80, 80, 3)
    camera = target["target_infos"][0]["smplx_params"]["K_intrinsic"][0]
    torch.testing.assert_close(camera[:2, 2], torch.tensor([39.5, 39.5]))
    scale = min(65 / driving_size[0], 79 / driving_size[1])
    torch.testing.assert_close(camera[0, 0], torch.tensor(50 * scale))
    image = target["target_video"][0]
    assert image[:, :7].count_nonzero() == 0 and image[:, 72:].count_nonzero() == 0
    assert image[79:].count_nonzero() == 0 and image.max() > 0
    dataset.close_all_engines()
