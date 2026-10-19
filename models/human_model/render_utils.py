import math
import os

import cv2
import numpy as np

os.environ["PYOPENGL_PLATFORM"] = "egl"
import torch
from pytorch3d.renderer import (
    Materials,
    MeshRasterizer,
    MeshRendererWithFragments,
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes


class Mesh_Render:
    def __init__(self, face):
        self.face = face.to(torch.long)[None, :, :]
        self.get_wall()

    def render(self, mesh, cam_param, bkg):
        return self.render_mesh(mesh, self.face, cam_param, bkg)

    def get_wall(self, wall_depth=2.5, wall_scale=1.25):
        self.wall_verts_template = torch.tensor(
            [
                [-wall_scale, wall_scale, wall_depth],
                [wall_scale, wall_scale, wall_depth],
                [wall_scale, -wall_scale, wall_depth],
                [-wall_scale, -wall_scale, wall_depth],
            ],
            device="cpu",
        ).float()

        w_x = self.wall_verts_template[:, 0]
        r_channel = (w_x + wall_scale) / (2.0 * wall_scale)
        w_y = self.wall_verts_template[:, 1]
        g_channel = (w_y + wall_scale) / (2.0 * wall_scale)
        b_channel = torch.ones_like(r_channel)
        self.wall_colors = torch.stack([r_channel, g_channel, b_channel], dim=1)
        self.wall_faces_template = torch.tensor([[0, 2, 1], [0, 3, 2]], device="cpu").long()

    def render_mesh(
        self, mesh, cam_param, bkg=0.0, down_sample=1.0, verts_features=None, with_wall=False
    ):
        batch_size, vertex_num = mesh.shape[:2]
        face = self.face.to(mesh.device).repeat(batch_size, 1, 1)
        render_shape = (
            int(cam_param["image_size"][1] / down_sample),
            int(cam_param["image_size"][0] / down_sample),
        )
        if "w2c_RT" in cam_param:
            mesh = torch.cat([mesh, torch.ones_like(mesh[:, :, :1])], dim=2)
            mesh = torch.einsum(
                "bij,bnj->bni",
                cam_param["w2c_RT"],
                mesh,
            )
            mesh = mesh[:, :, :3]

        mesh = torch.stack((-mesh[:, :, 0], -mesh[:, :, 1], mesh[:, :, 2]), 2)
        if with_wall:
            wall_verts = self.wall_verts_template.unsqueeze(0).repeat(batch_size, 1, 1)
            wall_feats = self.wall_colors
            wall_faces = self.wall_faces_template.unsqueeze(0).repeat(batch_size, 1, 1) + vertex_num

            wall_verts = torch.cat(
                [wall_verts.to(mesh.device), torch.ones_like(wall_verts[:, :, :1].to(mesh.device))],
                dim=2,
            )
            wall_verts = torch.einsum(
                "bij,bnj->bni",
                cam_param["w2c_RT"],
                wall_verts,
            )
            wall_verts = wall_verts[:, :, :3]

            mesh = torch.cat([mesh, wall_verts.to(mesh.device)], dim=1)
            face = torch.cat([face, wall_faces.to(face.device)], dim=1)
            verts_features = torch.cat(
                [verts_features, wall_feats.to(verts_features.device)], dim=0
            )

        if verts_features is None:
            textures = TexturesVertex(
                verts_features=torch.ones((batch_size, vertex_num, 3)).float().to(mesh.device)
            )
        else:
            verts_features = verts_features.unsqueeze(0).repeat(batch_size, 1, 1)
            textures = TexturesVertex(verts_features=verts_features.float().to(mesh.device))

        mesh = Meshes(mesh, face, textures)
        cameras = PerspectiveCameras(
            focal_length=cam_param["focal_length"] / down_sample,
            principal_point=(
                cam_param["princpt"].detach().clone().to(mesh.device).reshape(batch_size, 2)
                / down_sample
            ),
            device=mesh.device,
            in_ndc=False,
            image_size=torch.LongTensor(render_shape)
            .to(mesh.device)
            .reshape(1, 2)
            .repeat(batch_size, 1),
        )
        raster_settings = RasterizationSettings(
            image_size=render_shape,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=True,
            max_faces_per_bin=face.shape[1],
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(
            mesh.device
        )
        lights = PointLights(device=mesh.device, location=[[0.0, 0.0, -10.0]])
        shader = SoftPhongShader(device=mesh.device, cameras=cameras, lights=lights)
        materials = Materials(device=mesh.device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)

        with torch.no_grad():
            renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)
            images, fragments = renderer(mesh, materials=materials)

        if down_sample > 1.0:
            images = torch.nn.functional.interpolate(
                images.permute(0, 3, 1, 2),
                scale_factor=down_sample,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)

        render = images[:, :, :, :3]
        alpha_images = images[:, ..., 3:]
        mask = alpha_images.expand(-1, -1, -1, 3) < 0.5
        render = render * (1.0 - mask.float()) + mask * bkg
        render = render.squeeze(0).cpu().numpy() * 255

        return render


def draw_landmarks(landmarks, image, color=(255, 0, 0), radius=1, thickness=-1, show_index=False):
    canvs_img = image.copy()
    for idx, lmk in enumerate(landmarks):
        canvs_img = cv2.circle(canvs_img, (int(lmk[0]), int(lmk[1])), radius, color, thickness)
        if show_index:
            cv2.putText(
                canvs_img,
                f"{idx}",
                (int(lmk[0]), int(lmk[1])),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                ((idx * 2) % 255, 145, (idx * 3) % 255),
                1,
            )
    return canvs_img


def perspective_projection(w2c_RT, cam_params, points, image_height, image_width):
    points_cam = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_cam = torch.einsum("bij,bnj->bni", w2c_RT, points_cam)
    if "K_intrinsic" in cam_params:
        projected = torch.einsum("bij,bnj->bni", cam_params["K_intrinsic"], points_cam[..., :3])
        return projected[..., :2] / (projected[..., 2:3] + 1e-7), points_cam
    focalxy = torch.cat(
        [cam_params["focal_px"][..., None, 0:1], cam_params["focal_py"][..., None, 0:1]], dim=-1
    )
    points_img = points_cam[..., :2] * focalxy / (points_cam[..., 2:3] + 1e-7)
    points_img[..., 0:1] = points_img[..., 0:1] + image_width / 2
    points_img[..., 1:2] = points_img[..., 1:2] + image_height / 2
    return points_img, points_cam


FACEMESH_LIPS = [
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308),
]
FACEMESH_LIPS_REINDEXED = [
    (72, 82),
    (82, 80),
    (80, 84),
    (84, 77),
    (77, 68),
    (68, 95),
    (95, 102),
    (102, 98),
    (98, 100),
    (100, 90),
    (72, 85),
    (85, 71),
    (71, 70),
    (70, 69),
    (69, 65),
    (65, 87),
    (87, 88),
    (88, 89),
    (89, 103),
    (103, 90),
    (73, 81),
    (81, 79),
    (79, 83),
    (83, 78),
    (78, 67),
    (67, 96),
    (96, 101),
    (101, 97),
    (97, 99),
    (99, 91),
    (73, 86),
    (86, 74),
    (74, 75),
    (75, 76),
    (76, 66),
    (66, 94),
    (94, 93),
    (93, 92),
    (92, 104),
    (104, 91),
]
FACEMESH_LEFT_EYE = [
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362),
]
FACEMESH_LEFT_EYE_REINDEXED = [
    (21, 20),
    (20, 33),
    (33, 23),
    (23, 24),
    (24, 25),
    (25, 26),
    (26, 27),
    (27, 22),
    (21, 35),
    (35, 32),
    (32, 31),
    (31, 30),
    (30, 29),
    (29, 28),
    (28, 34),
    (34, 22),
]

FACEMESH_LEFT_EYEBROW = [
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336),
]
FACEMESH_LEFT_EYEBROW_REINDEXED = [(0, 2), (2, 1), (1, 5), (5, 3), (7, 4), (4, 8), (8, 6), (6, 9)]

FACEMESH_RIGHT_EYE = [
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133),
]
FACEMESH_RIGHT_EYE_REINDEXED = [
    (37, 36),
    (36, 49),
    (49, 39),
    (39, 40),
    (40, 41),
    (41, 42),
    (42, 43),
    (43, 38),
    (37, 51),
    (51, 48),
    (48, 47),
    (47, 46),
    (46, 45),
    (45, 44),
    (44, 50),
    (50, 38),
]

FACEMESH_RIGHT_EYEBROW = [
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107),
]
FACEMESH_RIGHT_EYEBROW_REINDEXED = [
    (10, 12),
    (12, 11),
    (11, 15),
    (15, 13),
    (17, 14),
    (14, 18),
    (18, 16),
    (16, 19),
]

FACEMESH_RIGHT_IRIS = [(1, 2), (2, 3), (3, 4), (4, 1)]
FACEMESH_LEFT_IRIS = [(6, 7), (7, 8), (8, 9), (9, 6)]


def reindex_connections(connections, mapping, name="Unknown"):
    new_connections = []
    skipped_count = 0

    for p1, p2 in connections:
        if p1 in mapping and p2 in mapping:
            new_p1 = mapping[p1]
            new_p2 = mapping[p2]
            new_connections.append((new_p1, new_p2))
        else:
            skipped_count += 1

    if skipped_count > 0:
        print(f"{name}: omitted {skipped_count} connections outside the landmark set.")

    return new_connections


def draw_135_MP_keypoints(
    keypoints,
    face_keypoints,
    irs_keypoints,
    lmk_mp_indices,
    img_origin,
    confidence_threshold=0.5,
    stick_width=None,
):
    img = img_origin.copy()
    height, width = img.shape[:2]

    if stick_width is None:
        stick_width = max(1, int(min(height, width) / 200))

    hand_line_width = max(1, int(stick_width / 2))
    face_line_width = hand_line_width

    if keypoints.shape[1] == 2:
        confs = np.ones((keypoints.shape[0], 1))
        keypoints = np.hstack((keypoints, confs))
    if face_keypoints is not None and len(face_keypoints) > 0 and face_keypoints.shape[1] == 2:
        confs = np.ones((face_keypoints.shape[0], 1))
        face_keypoints = np.hstack((face_keypoints, confs))
    c_body = (128, 0, 128)
    c_l_shoulder = (255, 128, 0)
    c_r_shoulder = (0, 165, 255)

    c_l_arm_up = (255, 100, 0)
    c_l_arm_low = (255, 200, 0)

    c_r_arm_up = (0, 100, 255)
    c_r_arm_low = (0, 200, 255)

    c_l_hip = (255, 255, 0)
    c_r_hip = (0, 0, 255)

    c_l_leg_up = (200, 255, 0)
    c_l_leg_low = (100, 255, 0)

    c_r_leg_up = (0, 0, 200)
    c_r_leg_low = (0, 0, 150)

    c_foot = (0, 255, 0)

    body_limbs = []

    body_limbs.append(([0, 7], c_body))
    body_limbs.append(([0, 1], c_l_hip))
    body_limbs.append(([0, 2], c_r_hip))
    body_limbs.append(([7, 8], c_l_shoulder))
    body_limbs.append(([7, 9], c_r_shoulder))

    body_limbs.append(([8, 10], c_l_arm_up))
    body_limbs.append(([10, 12], c_l_arm_low))
    body_limbs.append(([9, 11], c_r_arm_up))
    body_limbs.append(([11, 13], c_r_arm_low))

    body_limbs.append(([1, 3], c_l_leg_up))
    body_limbs.append(([3, 5], c_l_leg_low))
    body_limbs.append(([2, 4], c_r_leg_up))
    body_limbs.append(([4, 6], c_r_leg_low))

    body_limbs.append(([5, 16], c_foot))
    body_limbs.append(([5, 14], c_foot))
    body_limbs.append(([5, 15], c_foot))
    body_limbs.append(([14, 15], c_foot))
    body_limbs.append(([6, 19], c_foot))
    body_limbs.append(([6, 17], c_foot))
    body_limbs.append(([6, 18], c_foot))
    body_limbs.append(([17, 18], c_foot))

    for (idx1, idx2), color in body_limbs:
        if idx1 >= len(keypoints) or idx2 >= len(keypoints):
            continue
        p1, p2 = keypoints[idx1], keypoints[idx2]
        if p1[2] < confidence_threshold or p2[2] < confidence_threshold:
            continue

        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        mean_x, mean_y = (x1 + x2) / 2, (y1 + y2) / 2
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y1 - y2, x1 - x2))

        polygon = cv2.ellipse2Poly(
            (int(mean_x), int(mean_y)), (int(length / 2), stick_width), int(angle), 0, 360, 1
        )
        dimmed_color = [int(c * 0.6) for c in color]
        cv2.fillConvexPoly(img, polygon, dimmed_color)

    c_l_thumb = [(0, 0, 180), (0, 0, 220), (0, 0, 255), (80, 80, 255)]
    c_l_index = [(0, 100, 0), (0, 180, 0), (0, 255, 0), (80, 255, 80)]
    c_l_middle = [(100, 0, 0), (180, 0, 0), (255, 0, 0), (255, 80, 80)]
    c_l_ring = [(0, 180, 180), (0, 220, 220), (0, 255, 255), (100, 255, 255)]
    c_l_pinky = [(180, 0, 180), (220, 0, 220), (255, 0, 255), (255, 100, 255)]

    hands_config = [
        (12, [25, 26, 27, 28], c_l_thumb),
        (12, [29, 30, 31, 32], c_l_index),
        (12, [33, 34, 35, 36], c_l_middle),
        (12, [37, 38, 39, 40], c_l_ring),
        (12, [41, 42, 43, 44], c_l_pinky),
        (13, [45, 46, 47, 48], c_l_thumb),
        (13, [49, 50, 51, 52], c_l_index),
        (13, [53, 54, 55, 56], c_l_middle),
        (13, [57, 58, 59, 60], c_l_ring),
        (13, [61, 62, 63, 64], c_l_pinky),
    ]

    for wrist_idx, f_idxs, f_colors in hands_config:
        p1, p2 = keypoints[wrist_idx], keypoints[f_idxs[0]]
        if p1[2] >= confidence_threshold and p2[2] >= confidence_threshold:
            cv2.line(
                img,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                f_colors[0],
                hand_line_width,
            )

        for i in range(len(f_idxs) - 1):
            p1, p2 = keypoints[f_idxs[i]], keypoints[f_idxs[i + 1]]
            if p1[2] >= confidence_threshold and p2[2] >= confidence_threshold:
                col = f_colors[i + 1] if i + 1 < len(f_colors) else f_colors[-1]
                cv2.line(
                    img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, hand_line_width
                )

    if face_keypoints is not None and len(face_keypoints) > 0:
        c_face_lips = (255, 255, 255)
        c_face_left_eyebrow = (255, 0, 0)
        c_face_right_eyebrow = (0, 0, 255)
        c_face_left_eye = (255, 0, 0)
        c_face_right_eye = (0, 0, 255)
        c_face_left_iris = (255, 255, 0)
        c_face_right_iris = (255, 0, 255)

        mp_render_list = [
            (FACEMESH_LIPS_REINDEXED, c_face_lips),
            (FACEMESH_LEFT_EYEBROW_REINDEXED, c_face_left_eyebrow),
            (FACEMESH_RIGHT_EYEBROW_REINDEXED, c_face_right_eyebrow),
            (FACEMESH_LEFT_EYE_REINDEXED, c_face_left_eye),
            (FACEMESH_RIGHT_EYE_REINDEXED, c_face_right_eye),
        ]
        mp_iris_render_list = [
            (FACEMESH_LEFT_IRIS, c_face_left_iris),
            (FACEMESH_RIGHT_IRIS, c_face_right_iris),
        ]
        for connections, color in mp_render_list:
            for idx1, idx2 in connections:
                if idx1 >= len(face_keypoints) or idx2 >= len(face_keypoints):
                    continue

                pt1 = face_keypoints[idx1]
                pt2 = face_keypoints[idx2]

                if pt1.shape[0] > 2 and pt1[2] < 0.2:
                    continue
                if pt2.shape[0] > 2 and pt2[2] < 0.2:
                    continue

                p1_coords = (int(pt1[0]), int(pt1[1]))
                p2_coords = (int(pt2[0]), int(pt2[1]))

                cv2.line(img, p1_coords, p2_coords, color, face_line_width)
        for connections, color in mp_iris_render_list:
            for idx1, idx2 in connections:
                if idx1 >= len(irs_keypoints) or idx2 >= len(irs_keypoints):
                    continue
                pt1 = irs_keypoints[idx1]
                pt2 = irs_keypoints[idx2]

                if pt1.shape[0] > 2 and pt1[2] < 0.2:
                    continue
                if pt2.shape[0] > 2 and pt2[2] < 0.2:
                    continue

                p1_coords = (int(pt1[0]), int(pt1[1]))
                p2_coords = (int(pt2[0]), int(pt2[1]))

                cv2.line(img, p1_coords, p2_coords, color, face_line_width)
    face_idxs_to_remove = {20, 21, 22, 23, 24}
    body_idxs = [i for i in range(65) if i not in face_idxs_to_remove]
    body_idxs = set(body_idxs)
    for i, kp in enumerate(keypoints):
        if i not in body_idxs:
            continue
        if i > 70:
            break
        if kp[2] < confidence_threshold:
            continue

        x, y = int(kp[0]), int(kp[1])
        color = (200, 200, 200)

        if i in {0, 1, 2, 7, 8, 9}:
            color = c_body
        elif 25 <= i <= 64:
            color = (255, 255, 255)

        radius = stick_width if i < 25 else max(1, int(stick_width / 1.5))
        cv2.circle(img, (x, y), radius, color, -1)

    return img
