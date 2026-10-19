import pytest
import torch

from models.utils.video_chunks import padded_length, temporal_reference


@pytest.mark.parametrize("count,expected", [(1, 9), (5, 9), (9, 9), (13, 13), (14, 17), (17, 17)])
def test_chunk_padding_covers_short_and_partial_videos(count, expected):
    assert padded_length(count, 9, 5) == expected


def test_third_chunk_uses_five_actual_frames_across_the_output_boundary():
    first = torch.arange(9.0).reshape(1, 1, 9, 1, 1)
    second = torch.arange(9.0, 13.0).reshape(1, 1, 4, 1, 1)
    assert temporal_reference([first], 5).flatten().tolist() == [4, 5, 6, 7, 8]
    assert temporal_reference([first, second], 5).flatten().tolist() == [8, 9, 10, 11, 12]
    assert temporal_reference([first, second], 5).shape == (1, 5, 1, 1)


def test_regular_clip_preserves_the_original_temporal_reference():
    first = torch.randn(1, 3, 77, 4, 4)
    second = torch.randn(1, 3, 72, 4, 4)
    torch.testing.assert_close(temporal_reference([first, second], 5), second[0, :, -5:])


def test_clip_length_must_exceed_overlap():
    with pytest.raises(ValueError):
        padded_length(13, 5, 5)


def test_single_frame_padding_preserves_independent_parameters():
    from models.utils.general_utils import inputs_padding

    original = {"pose": torch.zeros(1)}
    padded = inputs_padding([original], 9)
    assert len(padded) == 9
    padded[0]["pose"] += 1
    assert original["pose"].item() == 0
    assert all(frame["pose"].item() == 0 for frame in padded[1:])
