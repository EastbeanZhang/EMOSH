from types import SimpleNamespace

import torch

from models.utils.retarget import retarget_identity


class HumanModel:
    smplx = SimpleNamespace(smplx2flame_ind=torch.tensor([0, 1]))

    def __call__(self, pose, expression, identity):
        vertices = torch.zeros(1, 3, 3) + identity["shape"]
        vertices = vertices + identity.get("global_offset", 0)
        return {"vertices": vertices}


def test_retarget_preserves_motion_and_head_position_without_mutating_reference():
    source = {"id_params": {"shape": torch.tensor(2.0)}}
    targets = [
        {
            "smplx_params": {"pose": torch.tensor(0.2)},
            "flame_params": {},
            "id_params": {"shape": torch.tensor(1.0)},
        }
        for _ in range(2)
    ]
    poses = [x["smplx_params"]["pose"].clone() for x in targets]
    retarget_identity(source, targets, HumanModel())
    assert "global_offset" not in source["id_params"]
    assert targets[0]["id_params"] is not targets[1]["id_params"]
    for info, pose in zip(targets, poses):
        assert info["id_params"]["shape"].item() == 2.0
        torch.testing.assert_close(info["smplx_params"]["pose"], pose)
        vertices = HumanModel()(info["smplx_params"], {}, info["id_params"])["vertices"]
        torch.testing.assert_close(vertices.mean(), torch.tensor(1.0))
