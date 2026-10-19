import torch


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1))

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale


def get_view_matrix(R, t):
    device = R.device
    Rt = torch.cat((R, t.view(3, 1)), 1)
    b_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 4)
    view_matrix = torch.cat((Rt, b_row))
    return view_matrix


def get_proj_matrix(
    tanfov,
    device,
    z_near=0.01,
    z_far=100,
    z_sign=1.0,
):
    tanHalfFovY = tanfov[1]
    tanHalfFovX = tanfov[0]

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    z_sign = 1.0

    proj_matrix = torch.zeros(4, 4).float().to(device)
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
    proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return proj_matrix


def get_full_proj_matrix(w2c_cam, tanfov, intrinsics=None, image_size=None):
    assert len(w2c_cam.shape) == 2
    view_matrix = get_view_matrix(w2c_cam[:3, :3], w2c_cam[:3, 3]).transpose(0, 1).contiguous()
    proj_matrix = get_proj_matrix(tanfov, device=w2c_cam.device)
    if intrinsics is not None:
        width, height = image_size
        proj_matrix[0, 2] = 2 * intrinsics[0, 2] / width - 1
        proj_matrix[1, 2] = 2 * intrinsics[1, 2] / height - 1
    proj_matrix = proj_matrix.transpose(0, 1).contiguous()
    full_proj_matrix = (view_matrix.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)

    return view_matrix, full_proj_matrix
