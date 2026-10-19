"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
"""

# Modified from smplx code for FLAME
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lbs import Struct, batch_rodrigues, lbs, to_np, to_tensor, vertices2landmarks


def rot_mat_to_euler(rot_mats):
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(self, model_path, shape_params=300, expression_params=100):
        super(FLAME, self).__init__()
        flame_model_path = os.path.join(model_path, "generic_model.pkl")
        lmk_fan_embedding_path = os.path.join(model_path, "landmark_embedding.npy")
        lmk_mp_embedding_path = os.path.join(model_path, "mediapipe_landmark_embedding.npz")
        lmk_203_embedding_path = os.path.join(model_path, "203_landmark_embeding.npz")

        left_eyelid = np.load(os.path.join(model_path, "l_eyelid.npy"))
        right_eyelid = np.load(os.path.join(model_path, "r_eyelid.npy"))

        print("creating the FLAME Decoder")
        with open(flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        self.register_buffer(
            "faces_tensor", to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long)
        )
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype)
        )
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [shapedirs[:, :, :shape_params], shapedirs[:, :, 300 : 300 + expression_params]], 2
        )
        self.register_buffer("shapedirs", shapedirs)
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter("eye_pose", nn.Parameter(default_eyball_pose, requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter("neck_pose", nn.Parameter(default_neck_pose, requires_grad=False))

        lmk_embeddings = np.load(lmk_fan_embedding_path, allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer(
            "lmk_faces_idx", torch.tensor(lmk_embeddings["static_lmk_faces_idx"], dtype=torch.long)
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.tensor(lmk_embeddings["static_lmk_bary_coords"], dtype=self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx",
            torch.tensor(lmk_embeddings["dynamic_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            torch.tensor(lmk_embeddings["dynamic_lmk_bary_coords"], dtype=self.dtype),
        )
        self.register_buffer(
            "full_lmk_faces_idx",
            torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long),
        )
        self.register_buffer(
            "full_lmk_bary_coords",
            torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype),
        )

        self.register_buffer("l_eyelid", torch.from_numpy(left_eyelid).to(self.dtype)[None])
        self.register_buffer("r_eyelid", torch.from_numpy(right_eyelid).to(self.dtype)[None])

        lmk_embeddings_mp = np.load(lmk_mp_embedding_path)
        self.register_buffer(
            "mp_lmk_faces_idx",
            torch.from_numpy(lmk_embeddings_mp["lmk_face_idx"].astype("int32")).long(),
        )
        self.register_buffer(
            "mp_lmk_bary_coords", torch.from_numpy(lmk_embeddings_mp["lmk_b_coords"]).to(self.dtype)
        )
        self.lmk_mp_indices = lmk_embeddings_mp["landmark_indices"].tolist()

        lmk_embeddings_203 = np.load(lmk_203_embedding_path)
        self.register_buffer(
            "lmk_203_faces_idx",
            torch.from_numpy(lmk_embeddings_203["lmk_face_idx"].astype("int32")).long(),
        )
        self.register_buffer(
            "lmk_203_bary_coords",
            torch.from_numpy(lmk_embeddings_203["lmk_b_coords"]).to(self.dtype),
        )
        self.lmk_203_mouth_indices = lmk_embeddings_203["landmark_mouth_indices"]
        self.lmk_203_front_indices = lmk_embeddings_203["landmark_front_indices"]
        self.lmk_203_left_indices = lmk_embeddings_203["landmark_left_indices"]
        self.lmk_203_right_indices = lmk_embeddings_203["landmark_right_indices"]

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(
        self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords, neck_kin_chain, dtype=torch.float32
    ):
        """
        Selects the face contour depending on the reletive position of the head
        Input:
            vertices: N X num_of_vertices X 3
            pose: N X full pose
            dynamic_lmk_faces_idx: The list of contour face indexes
            dynamic_lmk_b_coords: The list of contour barycentric weights
            neck_kin_chain: The tree to consider for the relative rotation
            dtype: Data type
        return:
            The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = (
            torch.eye(3, device=pose.device, dtype=dtype)
            .unsqueeze_(dim=0)
            .expand(batch_size, -1, -1)
        )
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx, 0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords, 0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
        Calculates landmarks by barycentric interpolation
        Input:
            vertices: torch.tensor NxVx3, dtype = torch.float32
                The tensor of input vertices
            faces: torch.tensor (N*F)x3, dtype = torch.long
                The faces of the mesh
            lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                The tensor with the indices of the faces used to calculate the
                landmarks.
            lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                The tensor of barycentric coordinates that are used to interpolate
                the landmarks

        Returns:
            landmarks: torch.tensor NxLx3, dtype = torch.float32
                The coordinates of the landmarks for each mesh in the batch
        """
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = (
            torch.index_select(faces, 0, lmk_faces_idx.view(-1))
            .view(1, -1, 3)
            .view(batch_size, lmk_faces_idx.shape[1], -1)
        )

        lmk_faces += (
            torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(device=vertices.device)
            * num_verts
        )

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum("blfi,blf->bli", [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        return landmarks3d

    def get_landmarks(self, vertices, full_pose):
        batch_size = vertices.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain,
            dtype=self.dtype,
        )
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(
            vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        bz = batch_size
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.repeat(bz, 1),
            self.full_lmk_bary_coords.repeat(bz, 1, 1),
        )

        landmarksmp = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.mp_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.mp_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        landmarks203 = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.lmk_203_faces_idx.repeat(vertices.shape[0], 1),
            self.lmk_203_bary_coords.repeat(vertices.shape[0], 1, 1),
        )

        return {
            "lmk_fan": landmarks2d,
            "lmk_mp": landmarksmp,
            "lmk_203": landmarks203,
            "lmk_fan_3d": landmarks3d,
        }

    def forward(self, params_dict: dict, return_lmk=False):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        shape_params = params_dict.get("shape_params", None)
        expression_params = params_dict.get("expression_params", None)
        pose_params = params_dict.get("pose_params", None)
        eye_pose_params = params_dict.get("eye_pose_params", None)

        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [
                pose_params[:, :3],
                self.neck_pose.expand(batch_size, -1),
                pose_params[:, 3:],
                eye_pose_params,
            ],
            dim=1,
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(
            betas,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )
        result = {
            "vertices": vertices,
        }
        if return_lmk:
            result.update(self.get_landmarks(vertices, full_pose))

        return result

    def cal_lmk203_loss(self, lmk203_pred, lmk203_gt, pose_params):
        loss_1, loss_2, loss_3 = 0.0, 0.0, 0.0
        mask = pose_params[:, 1] < -0.05
        if mask.sum() > 0:
            SELECTED_IDS = torch.tensor(self.lmk_203_left_indices).to(lmk203_pred.device)
            loss_1 = F.l1_loss(
                lmk203_pred[mask][:, SELECTED_IDS, :], lmk203_gt[mask][:, SELECTED_IDS, :]
            ).mean()

        mask = pose_params[:, 1] > 0.05
        if mask.sum() > 0:
            SELECTED_IDS = torch.tensor(self.lmk_203_right_indices).to(lmk203_pred.device)
            loss_2 = F.l1_loss(
                lmk203_pred[mask][:, SELECTED_IDS, :], lmk203_gt[mask][:, SELECTED_IDS, :]
            ).mean()

        mask = pose_params[:, 1].abs() <= 0.05
        if mask.sum() > 0:
            SELECTED_IDS = torch.tensor(self.lmk_203_front_indices).to(lmk203_pred.device)
            loss_3 = F.l1_loss(
                lmk203_pred[mask][:, SELECTED_IDS, :], lmk203_gt[mask][:, SELECTED_IDS, :]
            ).mean()

        return (loss_1 + loss_2 + loss_3) / 3.0
