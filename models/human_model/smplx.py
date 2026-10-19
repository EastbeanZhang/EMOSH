# a license agreement with MPG or you get the right to use the computer
# Any use of the computer program without a valid license is prohibited and
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# for Intelligent Systems. All rights reserved.
# Contact: ps-license@tuebingen.mpg.de

"""
original from https://github.com/vchoutas/smplx
modified by Vassilis and Yao
"""

import os.path as osp
import pickle

import numpy as np
import torch
import torch.nn as nn

from .joint_names import J14_NAMES, SMPLX_names, part_indices
from .lbs import (
    JointsFromVerticesSelector,
    Struct,
    find_dynamic_lmk_idx_and_bcoords,
    lbs,
    to_np,
    to_tensor,
    vertices2landmarks,
)
from .vertex_ids import vertex_ids as VERTEX_IDS
from .vertex_joint_selector import VertexJointSelector

head_kin_chain = [15, 12, 9, 6, 3, 0]


class SMPLX(nn.Module):
    """
    SMPL-X (SMPL eXpressive) is a unified body model, with shape parameters
    trained jointly for the face, hands and body.
    SMPL-X uses standard vertex based linear blend skinning with learned
    corrective blend shapes, has N=10475 vertices and K=54 joints,
    which includes joints for the neck, jaw, eyeballs and fingers.
    """

    NUM_BODY_JOINTS = 23 - 2
    NUM_HAND_JOINTS = 15
    NUM_FACE_JOINTS = 3
    NUM_JOINTS = NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS + NUM_FACE_JOINTS
    EXPRESSION_SPACE_DIM = 100
    NECK_IDX = 12

    def __init__(
        self,
        model_path: str,
        num_shape_params=300,
        num_expression_params=50,
    ) -> None:
        """SMPLX model constructor"""
        super(SMPLX, self).__init__()
        self.dtype = torch.float32
        self.num_shape_params = num_shape_params
        self.num_expression_params = num_expression_params
        smplx_model_path = osp.join(model_path, "SMPLX_NEUTRAL_2020.npz")
        extra_joint_path = osp.join(model_path, "smplx_extra_joints.yaml")
        j14_regressor_path = osp.join(model_path, "SMPLX_to_J14.pkl")
        model_data = np.load(smplx_model_path, allow_pickle=True)
        data_struct = Struct(**model_data)
        self.vertex_joint_selector = VertexJointSelector(vertex_ids=VERTEX_IDS["smplx"])

        self.register_buffer(
            "faces_tensor", to_tensor(to_np(data_struct.f, dtype=np.int64), dtype=torch.long)
        )
        self.register_buffer(
            "flame_faces_tensor", to_tensor(to_np(data_struct.f, dtype=np.int64), dtype=torch.long)
        )
        self.register_buffer(
            "v_template", to_tensor(to_np(data_struct.v_template), dtype=self.dtype)
        )

        shapedirs = to_tensor(to_np(data_struct.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat(
            [
                shapedirs[:, :, :num_shape_params],
                shapedirs[:, :, 300 : 300 + num_expression_params],
            ],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)
        num_pose_basis = data_struct.posedirs.shape[-1]
        posedirs = np.reshape(data_struct.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer(
            "J_regressor", to_tensor(to_np(data_struct.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer("lbs_weights", to_tensor(to_np(data_struct.weights), dtype=self.dtype))

        self.register_buffer(
            "lmk_faces_idx", torch.tensor(data_struct.lmk_faces_idx, dtype=torch.long)
        )
        self.register_buffer(
            "lmk_bary_coords",
            torch.tensor(data_struct.lmk_bary_coords, dtype=self.dtype),
        )
        self.register_buffer(
            "dynamic_lmk_faces_idx",
            torch.tensor(data_struct.dynamic_lmk_faces_idx, dtype=torch.long),
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords",
            torch.tensor(data_struct.dynamic_lmk_bary_coords, dtype=self.dtype),
        )

        self.register_buffer("head_kin_chain", torch.tensor(head_kin_chain, dtype=torch.long))

        self.register_buffer(
            "shape_params",
            nn.Parameter(torch.zeros([1, num_shape_params], dtype=self.dtype), requires_grad=False),
        )
        self.register_buffer(
            "expression_params",
            nn.Parameter(
                torch.zeros([1, num_expression_params], dtype=self.dtype), requires_grad=False
            ),
        )

        self.register_buffer(
            "global_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "head_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "neck_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "jaw_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(1, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "eye_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(2, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "body_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(21, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "left_hand_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15, 1, 1),
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "right_hand_pose",
            nn.Parameter(
                torch.eye(3, dtype=self.dtype).unsqueeze(0).repeat(15, 1, 1),
                requires_grad=False,
            ),
        )

        self.extra_joint_selector = JointsFromVerticesSelector(extra_joint_path)
        self.use_joint_regressor = True
        self.keypoint_names = SMPLX_names
        if self.use_joint_regressor:
            with open(j14_regressor_path, "rb") as f:
                j14_regressor = pickle.load(f, encoding="latin1")
            source = []
            target = []
            for idx, name in enumerate(self.keypoint_names):
                if name in J14_NAMES:
                    source.append(idx)
                    target.append(J14_NAMES.index(name))
            source = np.asarray(source)
            target = np.asarray(target)
            self.register_buffer("source_idxs", torch.from_numpy(source))
            self.register_buffer("target_idxs", torch.from_numpy(target))
            self.register_buffer(
                "extra_joint_regressor", torch.from_numpy(j14_regressor).to(torch.float32)
            )
            self.part_indices = part_indices

        self.smplx2flame_ind = np.load(osp.join(model_path, "SMPL-X__FLAME_vertex_ids.npy"))
        with open(osp.join(model_path, "MANO_SMPLX_vertex_ids.pkl"), "rb") as file:
            self.smplx2mano_ind = pickle.load(file, encoding="latin1")
        self.get_joints_kpt_info()

        left_hand_center = self.v_template[self.smplx2mano_ind["left_hand"], :].mean(0)
        right_hand_center = self.v_template[self.smplx2mano_ind["right_hand"], :].mean(0)
        self.register_buffer("left_hand_center", left_hand_center)
        self.register_buffer("right_hand_center", right_hand_center)

    def name(self) -> str:
        return "SMPLX"

    def create_mean_pose(self, data_struct, flat_hand_mean=False):
        global_orient_mean = torch.zeros([3], dtype=self.dtype)
        body_pose_mean = torch.zeros([self.NUM_BODY_JOINTS * 3], dtype=self.dtype)
        jaw_pose_mean = torch.zeros([3], dtype=self.dtype)
        leye_pose_mean = torch.zeros([3], dtype=self.dtype)
        reye_pose_mean = torch.zeros([3], dtype=self.dtype)

        pose_mean = np.concatenate(
            [
                global_orient_mean,
                body_pose_mean,
                jaw_pose_mean,
                leye_pose_mean,
                reye_pose_mean,
                self.left_hand_mean,
                self.right_hand_mean,
            ],
            axis=0,
        )

        return pose_mean

    def forward(
        self,
        params_dict: dict,
        return_landmarks: bool = True,
    ):
        """
        Args:
            shape_params: [N, number of shape parameters]
            expression_params: [N, number of expression parameters]
            global_pose: pelvis pose, [N, 1, 3, 3]
            body_pose: [N, 21, 3, 3]
            jaw_pose: [N, 1, 3, 3]
            eye_pose: [N, 2, 3, 3]
            left_hand_pose: [N, 15, 3, 3]
            right_hand_pose: [N, 15, 3, 3]
        Returns:
            vertices: [N, number of vertices, 3]
            landmarks: [N, number of landmarks (68 face keypoints), 3]
            joints: [N, number of smplx joints (145), 3]
        """
        shape_params = params_dict.get("shape_params", None)
        expression_params = params_dict.get("expression_params", None)
        global_pose = params_dict.get("global_pose", None)
        body_pose = params_dict.get("body_pose", None)
        jaw_pose = params_dict.get("jaw_pose", None)
        eye_pose = params_dict.get("eye_pose", None)
        left_hand_pose = params_dict.get("left_hand_pose", None)
        right_hand_pose = params_dict.get("right_hand_pose", None)

        if shape_params is None:
            batch_size = global_pose.shape[0]
            shape_params = self.shape_params.expand(batch_size, -1)
        else:
            batch_size = shape_params.shape[0]
        if expression_params is None:
            expression_params = self.expression_params.expand(batch_size, -1)
        if global_pose is None:
            global_pose = self.global_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if body_pose is None:
            body_pose = self.body_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if jaw_pose is None:
            jaw_pose = self.jaw_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if eye_pose is None:
            eye_pose = self.eye_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if left_hand_pose is None:
            left_hand_pose = self.left_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if right_hand_pose is None:
            right_hand_pose = self.right_hand_pose.unsqueeze(0).expand(batch_size, -1, -1, -1)

        shape_components = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [
                global_pose,
                body_pose,
                jaw_pose,
                eye_pose,
                left_hand_pose,
                right_hand_pose,
            ],
            dim=1,
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, joints = lbs(
            shape_components,
            full_pose,
            template_vertices,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
            pose2rot=False,
        )
        result = {
            "vertices": vertices,
            "full_pose": full_pose,
            "ori_joints": joints,
        }
        if return_landmarks:
            landmarks, joints = self.get_landkmarks(full_pose, vertices, joints)
            result["landmarks"] = landmarks
            result["joints"] = joints

        return result

    def get_landkmarks(self, full_pose, vertices, joints):
        batch_size = full_pose.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
            vertices,
            full_pose,
            self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.head_kin_chain,
        )
        lmk_faces_idx = torch.cat([lmk_faces_idx, dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([lmk_bary_coords, dyn_lmk_bary_coords], 1)
        landmarks = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        joints = self.vertex_joint_selector(vertices, joints)
        final_joint_set = [joints, landmarks]
        joints = torch.cat(final_joint_set, dim=1)
        return landmarks, joints

    def pose_abs2rel(self, global_pose, body_pose, abs_joint="head"):
        """change absolute pose to relative pose
        Basic knowledge for SMPLX kinematic tree:
                absolute pose = parent pose * relative pose
        Here, pose must be represented as rotation matrix (batch_sizexnx3x3)
        """
        if abs_joint == "head":
            kin_chain = [15, 12, 9, 6, 3, 0]
        elif abs_joint == "neck":
            kin_chain = [12, 9, 6, 3, 0]
        elif abs_joint == "right_wrist":
            kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
        elif abs_joint == "left_wrist":
            kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
        else:
            raise NotImplementedError(f"pose_abs2rel does not support: {abs_joint}")

        batch_size = global_pose.shape[0]
        dtype = global_pose.dtype
        device = global_pose.device
        full_pose = torch.cat([global_pose, body_pose], dim=1)
        rel_rot_mat = (
            torch.eye(3, device=device, dtype=dtype).unsqueeze_(dim=0).repeat(batch_size, 1, 1)
        )
        for idx in kin_chain[1:]:
            rel_rot_mat = torch.bmm(full_pose[:, idx], rel_rot_mat)

        abs_parent_pose = rel_rot_mat.detach()
        abs_joint_pose = body_pose[:, kin_chain[0] - 1]
        rel_joint_pose = torch.matmul(
            abs_parent_pose.reshape(-1, 3, 3).transpose(1, 2),
            abs_joint_pose.reshape(-1, 3, 3),
        )
        body_pose[:, kin_chain[0] - 1, :, :] = rel_joint_pose
        return body_pose

    def pose_rel2abs(self, global_pose, body_pose, abs_joint="head"):
        """change relative pose to absolute pose
        Basic knowledge for SMPLX kinematic tree:
                absolute pose = parent pose * relative pose
        Here, pose must be represented as rotation matrix (batch_sizexnx3x3)
        """
        full_pose = torch.cat([global_pose, body_pose], dim=1)

        if abs_joint == "head":
            kin_chain = [15, 12, 9, 6, 3, 0]
        elif abs_joint == "neck":
            kin_chain = [12, 9, 6, 3, 0]
        elif abs_joint == "right_wrist":
            kin_chain = [21, 19, 17, 14, 9, 6, 3, 0]
        elif abs_joint == "left_wrist":
            kin_chain = [20, 18, 16, 13, 9, 6, 3, 0]
        else:
            raise NotImplementedError(f"pose_rel2abs does not support: {abs_joint}")
        rel_rot_mat = torch.eye(3, device=full_pose.device, dtype=full_pose.dtype).unsqueeze_(dim=0)
        for idx in kin_chain:
            rel_rot_mat = torch.matmul(full_pose[:, idx], rel_rot_mat)
        abs_pose = rel_rot_mat[:, None, :, :]
        return abs_pose

    def get_joints_kpt_info(self):
        self.joint_dict = {
            "num": 55,
            "name": (
                "Pelvis",
                "L_Hip",
                "R_Hip",
                "Spine_1",
                "L_Knee",
                "R_Knee",
                "Spine_2",
                "L_Ankle",
                "R_Ankle",
                "Spine_3",
                "L_Foot",
                "R_Foot",
                "Neck",
                "L_Collar",
                "R_Collar",
                "Head",
                "L_Shoulder",
                "R_Shoulder",
                "L_Elbow",
                "R_Elbow",
                "L_Wrist",
                "R_Wrist",
                "Jaw",
                "L_Eye",
                "R_Eye",
                "L_Index_1",
                "L_Index_2",
                "L_Index_3",
                "L_Middle_1",
                "L_Middle_2",
                "L_Middle_3",
                "L_Pinky_1",
                "L_Pinky_2",
                "L_Pinky_3",
                "L_Ring_1",
                "L_Ring_2",
                "L_Ring_3",
                "L_Thumb_1",
                "L_Thumb_2",
                "L_Thumb_3",
                "R_Index_1",
                "R_Index_2",
                "R_Index_3",
                "R_Middle_1",
                "R_Middle_2",
                "R_Middle_3",
                "R_Pinky_1",
                "R_Pinky_2",
                "R_Pinky_3",
                "R_Ring_1",
                "R_Ring_2",
                "R_Ring_3",
                "R_Thumb_1",
                "R_Thumb_2",
                "R_Thumb_3",
            ),
        }
        self.joint_dict["root_idx"] = self.joint_dict["name"].index("Pelvis")
        self.joint_dict["part_idx"] = {
            "body": range(
                self.joint_dict["name"].index("Pelvis"),
                self.joint_dict["name"].index("R_Wrist") + 1,
            ),
            "face": range(
                self.joint_dict["name"].index("Jaw"), self.joint_dict["name"].index("R_Eye") + 1
            ),
            "lhand": range(
                self.joint_dict["name"].index("L_Index_1"),
                self.joint_dict["name"].index("L_Thumb_3") + 1,
            ),
            "rhand": range(
                self.joint_dict["name"].index("R_Index_1"),
                self.joint_dict["name"].index("R_Thumb_3") + 1,
            ),
        }

        self.kpt_dict = {
            "num": 135,
            "name": (
                "Pelvis",
                "L_Hip",
                "R_Hip",
                "L_Knee",
                "R_Knee",
                "L_Ankle",
                "R_Ankle",
                "Neck",
                "L_Shoulder",
                "R_Shoulder",
                "L_Elbow",
                "R_Elbow",
                "L_Wrist",
                "R_Wrist",
                "L_Big_toe",
                "L_Small_toe",
                "L_Heel",
                "R_Big_toe",
                "R_Small_toe",
                "R_Heel",
                "L_Ear",
                "R_Ear",
                "L_Eye",
                "R_Eye",
                "Nose",
                "L_Thumb_1",
                "L_Thumb_2",
                "L_Thumb_3",
                "L_Thumb_4",
                "L_Index_1",
                "L_Index_2",
                "L_Index_3",
                "L_Index_4",
                "L_Middle_1",
                "L_Middle_2",
                "L_Middle_3",
                "L_Middle_4",
                "L_Ring_1",
                "L_Ring_2",
                "L_Ring_3",
                "L_Ring_4",
                "L_Pinky_1",
                "L_Pinky_2",
                "L_Pinky_3",
                "L_Pinky_4",
                "R_Thumb_1",
                "R_Thumb_2",
                "R_Thumb_3",
                "R_Thumb_4",
                "R_Index_1",
                "R_Index_2",
                "R_Index_3",
                "R_Index_4",
                "R_Middle_1",
                "R_Middle_2",
                "R_Middle_3",
                "R_Middle_4",
                "R_Ring_1",
                "R_Ring_2",
                "R_Ring_3",
                "R_Ring_4",
                "R_Pinky_1",
                "R_Pinky_2",
                "R_Pinky_3",
                "R_Pinky_4",
                "Head",
                "Jaw",
                *["Face_" + str(i) for i in range(1, 69)],
            ),
            "idx": (
                0,
                1,
                2,
                4,
                5,
                7,
                8,
                12,
                16,
                17,
                18,
                19,
                20,
                21,
                60,
                61,
                62,
                63,
                64,
                65,
                59,
                58,
                57,
                56,
                55,
                37,
                38,
                39,
                66,
                25,
                26,
                27,
                67,
                28,
                29,
                30,
                68,
                34,
                35,
                36,
                69,
                31,
                32,
                33,
                70,
                52,
                53,
                54,
                71,
                40,
                41,
                42,
                72,
                43,
                44,
                45,
                73,
                49,
                50,
                51,
                74,
                46,
                47,
                48,
                75,
                15,
                22,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
                121,
                122,
                123,
                124,
                125,
                126,
                127,
                128,
                129,
                130,
                131,
                132,
                133,
                134,
                135,
                136,
                137,
                138,
                139,
                140,
                141,
                142,
                143,
            ),
        }
        self.kpt_dict["root_idx"] = self.kpt_dict["name"].index("Pelvis")
        self.kpt_dict["part_idx"] = {
            "body": range(
                self.kpt_dict["name"].index("Pelvis"), self.kpt_dict["name"].index("Nose") + 1
            ),
            "lhand": range(
                self.kpt_dict["name"].index("L_Thumb_1"),
                self.kpt_dict["name"].index("L_Pinky_4") + 1,
            ),
            "rhand": range(
                self.kpt_dict["name"].index("R_Thumb_1"),
                self.kpt_dict["name"].index("R_Pinky_4") + 1,
            ),
            "face": [
                self.kpt_dict["name"].index("Neck"),
                self.kpt_dict["name"].index("Head"),
                self.kpt_dict["name"].index("Jaw"),
                self.kpt_dict["name"].index("L_Eye"),
                self.kpt_dict["name"].index("R_Eye"),
            ]
            + list(
                range(
                    self.kpt_dict["name"].index("Face_1"),
                    self.kpt_dict["name"].index("Face_68") + 1,
                )
            )
            + [self.kpt_dict["name"].index("L_Ear"), self.kpt_dict["name"].index("R_Ear")],
        }
