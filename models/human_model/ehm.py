import os

import torch
import torch.nn as nn

from .flame import FLAME
from .lbs import (
    blend_shapes,
    find_dynamic_lmk_idx_and_bcoords,
    lbs_shaped,
    vertices2joints,
    vertices2landmarks,
)
from .mano import MANO
from .smplx import SMPLX


class EHM(nn.Module):
    """
    EHM (eXpressive human) is a unified body model, combined with flame and smplx.
    """

    def __init__(
        self,
        model_path: str,
    ) -> None:
        """SMPLX model constructor"""
        super(EHM, self).__init__()

        self.smplx = SMPLX(
            model_path=os.path.join(model_path, "smplx"),
            num_shape_params=300,
            num_expression_params=0,
        )
        self.flame = FLAME(
            model_path=os.path.join(model_path, "flame"),
            shape_params=300,
            expression_params=50,
        )
        self.mano = MANO(model_path=os.path.join(model_path, "mano"))

        template_joints = vertices2joints(
            self.flame.J_regressor, self.flame.v_template.unsqueeze(0)
        )
        self.flame_eyes_joints = template_joints[0, -2:, :]
        template_joints = vertices2joints(
            self.smplx.J_regressor, self.smplx.v_template.unsqueeze(0)
        )
        self.smplx_eyes_joints = template_joints[0, 23:25, :]
        eyes_offset = (self.smplx_eyes_joints - self.flame_eyes_joints).mean(dim=0, keepdim=True)

        v_template, v_head_template = self.smplx.v_template.clone(), self.flame.v_template.clone()
        v_template[self.smplx.smplx2flame_ind] = v_head_template + eyes_offset
        self.register_buffer("v_template", v_template)
        self.eye_mp_idx = [4597, 4543, 4511, 4479, 4575, 4051, 3997, 3965, 3933, 4029]

    def forward(
        self,
        smplx_param_dict: dict,
        flame_param_dict: dict,
        id_param_dict: dict,
        return_landmarks=False,
    ):
        eyelid_params = flame_param_dict.get("eyelid_params", None)
        expression_params = flame_param_dict["expression_params"]
        batch_size = expression_params.shape[0]

        hand_scale = id_param_dict.get("hand_scale", None)
        head_scale = id_param_dict.get("head_scale", None)
        head_shape_params = id_param_dict["flame_shape"]
        body_shape_params = id_param_dict["smplx_shape"]
        joints_offset = id_param_dict["joints_offset"]
        global_offset = id_param_dict.get("global_offset", None)

        head_scale = head_scale.expand(batch_size, -1, -1) if head_scale is not None else None
        hand_scale = hand_scale.expand(batch_size, -1, -1) if hand_scale is not None else None
        head_shape_params = head_shape_params.expand(batch_size, -1)
        body_shape_params = body_shape_params.expand(batch_size, -1)

        joints_offset = joints_offset.expand(batch_size, -1, -1)

        if head_scale is None:
            head_scale = 1.0
        betas = torch.cat([head_shape_params, expression_params], dim=1)
        head_template_vertices = self.flame.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        head_shaped = head_template_vertices + blend_shapes(betas, self.flame.shapedirs)
        head_shaped = head_shaped * head_scale
        head_joints = vertices2joints(self.flame.J_regressor, head_shaped)
        if eyelid_params is not None:
            head_shaped = (
                head_shaped
                + self.flame.r_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 1:2, None]
            )
            head_shaped = (
                head_shaped
                + self.flame.l_eyelid.expand(batch_size, -1, -1) * eyelid_params[:, 0:1, None]
            )

        global_pose = smplx_param_dict["global_pose"]
        body_pose = smplx_param_dict["body_pose"]
        jaw_pose = smplx_param_dict["jaw_pose"]
        left_hand_pose = smplx_param_dict["left_hand_pose"]
        right_hand_pose = smplx_param_dict["right_hand_pose"]
        eye_pose = smplx_param_dict["eye_pose"]

        shape_components = body_shape_params
        full_pose = torch.cat(
            [global_pose, body_pose, jaw_pose, eye_pose, left_hand_pose, right_hand_pose], dim=1
        )

        human_template_vertices = self.smplx.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        human_shaped_vertices = human_template_vertices + blend_shapes(
            shape_components, self.smplx.shapedirs
        )
        tbody_joints = vertices2joints(self.smplx.J_regressor, human_shaped_vertices)
        if joints_offset is not None:
            tbody_joints = tbody_joints + joints_offset

        eyes_offset = (tbody_joints[:, 23:25, :] - head_joints[:, -2:, :]).mean(dim=1, keepdim=True)
        head_shaped = head_shaped + eyes_offset
        head_joints = head_joints + eyes_offset
        tbody_joints[:, 22] = head_joints[:, 2]
        human_shaped_vertices[:, self.smplx.smplx2flame_ind] = head_shaped

        if hand_scale is not None:
            left_hand_ver = human_shaped_vertices[:, self.smplx.smplx2mano_ind["left_hand"]].clone()
            right_hand_ver = human_shaped_vertices[
                :, self.smplx.smplx2mano_ind["right_hand"]
            ].clone()
            right_hand_ver = (
                right_hand_ver * hand_scale[:, 1:2]
                + (1 - hand_scale[:, 1:2]) * self.smplx.right_hand_center[None, None]
            )
            left_hand_ver = (
                left_hand_ver * hand_scale[:, 0:1]
                + (1 - hand_scale[:, 0:1]) * self.smplx.left_hand_center[None, None]
            )
            human_shaped_vertices[:, self.smplx.smplx2mano_ind["right_hand"]] = right_hand_ver
            human_shaped_vertices[:, self.smplx.smplx2mano_ind["left_hand"]] = left_hand_ver

        vertices, joints = lbs_shaped(
            full_pose,
            human_shaped_vertices,
            self.smplx.posedirs,
            self.smplx.J_regressor,
            self.smplx.parents,
            self.smplx.lbs_weights,
            tbody_joints,
        )

        if global_offset is not None:
            vertices = vertices + global_offset
            joints = joints + global_offset
        result = {
            "vertices": vertices,
            "joints": joints,
            "full_pose": full_pose,
        }

        if return_landmarks:
            landmarks = self.get_landmarks(vertices, joints, full_pose)
            result.update(landmarks)

        return result

    def get_landmarks(self, vertices, joints, full_pose):
        batch_size = vertices.shape[0]
        lmk_faces_idx = self.smplx.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.smplx.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
            vertices,
            full_pose,
            self.smplx.dynamic_lmk_faces_idx,
            self.smplx.dynamic_lmk_bary_coords,
            self.smplx.head_kin_chain,
        )
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        landmarks = vertices2landmarks(
            vertices, self.smplx.faces_tensor, lmk_faces_idx, lmk_bary_coords
        )
        joints = self.smplx.vertex_joint_selector(vertices, joints)

        final_joint_set = [joints, landmarks]
        joints = torch.cat(final_joint_set, dim=1)

        head_vertices = vertices[:, self.smplx.smplx2flame_ind]
        landmarks_mp = vertices2landmarks(
            head_vertices,
            self.flame.faces_tensor,
            self.flame.mp_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.flame.mp_lmk_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        landmarks_203 = vertices2landmarks(
            head_vertices,
            self.flame.faces_tensor,
            self.flame.lmk_203_faces_idx.repeat(vertices.shape[0], 1),
            self.flame.lmk_203_bary_coords.repeat(vertices.shape[0], 1, 1),
        )
        landmarks_whole = joints[:, self.smplx.kpt_dict["idx"], :]
        results = {
            "landmarks_whole": landmarks_whole,
            "landmarks_face": landmarks,
            "landmarks_mp": landmarks_mp,
            "landmarks_203": landmarks_203,
            "joints": joints,
        }

        return results
