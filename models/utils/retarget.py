from copy import deepcopy

import torch


@torch.no_grad()
def retarget_identity(source_info, target_infos, ehm):
    identity = deepcopy(source_info["id_params"])
    first = target_infos[0]
    original = ehm(first["smplx_params"], first["flame_params"], first["id_params"])
    retargeted = ehm(first["smplx_params"], first["flame_params"], identity)
    indices = ehm.smplx.smplx2flame_ind
    offset = original["vertices"][:, indices].mean(1, keepdim=True)
    offset = offset - retargeted["vertices"][:, indices].mean(1, keepdim=True)
    identity["global_offset"] = identity.get("global_offset", 0) + offset
    for information in target_infos:
        information["id_params"] = deepcopy(identity)
    return target_infos
