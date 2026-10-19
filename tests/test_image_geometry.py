from copy import deepcopy

import pytest
import torch

from models.utils.graphics_utils import get_full_proj_matrix
from models.utils.image_geometry import canvas_size, fit_canvas


@pytest.mark.parametrize(
    "width,height", [(1920, 1080), (1080, 1920), (513, 769), (769, 513), (64, 4096)]
)
def test_native_canvas_and_area_budget(width, height):
    w, h = canvas_size(width, height)
    assert w >= width and h >= height and w % 16 == h % 16 == 0
    w, h = canvas_size(width, height, resolution_area=(1280, 720))
    assert min(w, h) >= 64 and w % 16 == h % 16 == 0 and w * h <= 1280 * 720
    if max(width / height, height / width) < 10:
        assert abs(w / h - width / height) / (width / height) < 0.04


@pytest.mark.parametrize("size", [(63, 128), (127, 256), (128, 0)])
def test_invalid_canvas_is_rejected(size):
    with pytest.raises(ValueError):
        canvas_size(640, 480, size=size)


def tracking(width, height):
    intrinsic = torch.tensor([[420.0, 0, width / 2 - 7], [0, 430.0, height / 2 + 3], [0, 0, 1]])
    return {
        "smplx_params": {
            "K_intrinsic": intrinsic,
            "K_original": intrinsic.clone(),
            "focal_px": torch.tensor([420.0]),
            "focal_py": torch.tensor([430.0]),
        },
        "full_crop": {"M_o2c-full": torch.eye(3), "M_c2o-full": torch.eye(3)},
        "head_crop": {"M_c2o": torch.tensor([[0.3, 0, 21], [0, 0.3, 17], [0, 0, 1.0]])},
    }


@pytest.mark.parametrize("size", [(384, 128), (128, 384)])
def test_resizing_keeps_camera_and_crop_projection_aligned(size):
    width, height = 192, 112
    info = tracking(width, height)
    original = deepcopy(info)
    image = torch.zeros(3, height, width)
    image[:, 56, 96] = 1
    fitted, mask = fit_canvas(image, info, size)
    transform = info["full_crop"]["M_o2c-full"]
    points = torch.tensor([[0.1, 0.2, 2.0], [-0.2, 0.1, 3.0]]).T
    old = original["smplx_params"]["K_intrinsic"] @ points
    new = info["smplx_params"]["K_intrinsic"] @ points
    torch.testing.assert_close(new, transform @ old)
    torch.testing.assert_close(
        info["smplx_params"]["K_original"], original["smplx_params"]["K_original"]
    )
    torch.testing.assert_close(
        info["full_crop"]["M_c2o-full"] @ transform, torch.eye(3), atol=1e-5, rtol=1e-5
    )
    corners = torch.tensor([[0.0, 256, 256, 0], [0, 0, 256, 256], [1, 1, 1, 1]])
    torch.testing.assert_close(
        transform @ original["head_crop"]["M_c2o"] @ corners,
        info["full_crop"]["M_o2c-full"] @ info["head_crop"]["M_c2o"] @ corners,
    )
    location = transform @ torch.tensor([96.0, 56.0, 1.0])
    assert fitted[:, round(location[1].item()), round(location[0].item())].max() > 0.2
    assert fitted.shape == (3, size[1], size[0]) and mask.shape == fitted.shape


def test_odd_native_size_preserves_pixels_and_off_center_camera():
    width, height = 65, 79
    image = torch.rand(3, height, width)
    info = tracking(width, height)
    original = deepcopy(info)
    fitted, _ = fit_canvas(image, info, canvas_size(width, height), pad_only=True)
    torch.testing.assert_close(fitted[:, 0:79, 7:72], image, atol=1e-5, rtol=1e-5)
    expected = original["smplx_params"]["K_intrinsic"].clone()
    expected[0, 2] += 7
    torch.testing.assert_close(info["smplx_params"]["K_intrinsic"], expected)


def test_projection_matrix_uses_recorded_principal_point():
    intrinsic = torch.tensor([[420.0, 0, 149], [0, 430.0, 77], [0, 0, 1]])
    _, proj = get_full_proj_matrix(
        torch.eye(4),
        torch.tensor([320 / 840, 192 / 860]),
        intrinsics=intrinsic,
        image_size=(320, 192),
    )
    point = torch.tensor([0.1, 0.2, 2.0, 1.0])
    projected = point @ proj
    ndc = projected[:2] / projected[3]
    pixels = (ndc + 1) * torch.tensor([320, 192]) / 2
    expected = intrinsic @ point[:3]
    torch.testing.assert_close(pixels, expected[:2] / expected[2])


@pytest.mark.gpu
@pytest.mark.parametrize("size", [(1280, 720), (720, 1280), (464, 640)])
def test_mesh_camera_matches_pixel_projection(size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from pytorch3d.renderer import PerspectiveCameras

    from models.human_model.render_utils import perspective_projection

    width, height = size
    intrinsic = torch.tensor(
        [[[650.0, 0, width / 2 - 3.5], [0, 660.0, height / 2 + 4.5], [0, 0, 1.0]]], device="cuda"
    )
    points = torch.tensor([[[0.1, 0.2, 2.0], [-0.3, 0.2, 3.0], [0.2, -0.1, 1.5]]], device="cuda")
    parameters = {"K_intrinsic": intrinsic}
    pixels, _ = perspective_projection(
        torch.eye(4, device="cuda")[None], parameters, points, height, width
    )
    cameras = PerspectiveCameras(
        focal_length=((650.0, 660.0),),
        principal_point=intrinsic[:, :2, 2],
        image_size=((height, width),),
        in_ndc=False,
        device="cuda",
    )
    flipped = points * points.new_tensor([-1, -1, 1])
    actual = cameras.transform_points_screen(flipped)[..., :2]
    torch.testing.assert_close(actual, pixels, atol=1e-3, rtol=1e-5)
