import argparse

import numpy as np
import torch

from models.utils.data_utils import Tracked_VideoDataset_infer
from models.utils.general_utils import save_video
from models.utils.image_geometry import add_resolution_arguments, validate_resolution_arguments
from models.utils.retarget import retarget_identity
from models.wan.configs import WAN_CONFIGS
from models.wan.wan_video_pipe import WanVideoPipeline


def main():
    p = argparse.ArgumentParser(
        description="Animate a tracked reference with tracked driving motion."
    )
    p.add_argument("--dataset", required=True)
    p.add_argument("--reference", required=True, help="Reference record ID in the prepared dataset")
    p.add_argument("--driving", required=True, help="Driving record ID in the prepared dataset")
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--checkpoint", required=True, help="A model.safetensors file produced by train.py"
    )
    p.add_argument("--assets", default="assets")
    p.add_argument("--output", default="outputs/animation.mp4")
    add_resolution_arguments(p)
    p.add_argument("--num-frames", type=int, default=77)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    anchor = p.add_mutually_exclusive_group()
    anchor.add_argument("--spatial-anchor", action="store_true", default=False)
    anchor.add_argument("--disable-spatial-anchor", dest="spatial_anchor", action="store_false")
    p.add_argument("--sample-solver", choices=["flowmatch", "unipc", "dpm++"], default="flowmatch")
    args = p.parse_args()
    validate_resolution_arguments(p, args)
    if args.num_frames < 9 or args.num_frames % 4 != 1:
        p.error("--num-frames must be 4n+1 and at least 9 for five-frame overlap")
    if args.steps <= 0 or (args.fps is not None and args.fps <= 0):
        p.error("--steps and --fps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires an NVIDIA CUDA GPU.")
    dataset = Tracked_VideoDataset_infer(
        base_path=args.dataset,
        split="test",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        resolution_area=args.resolution_area,
    )
    if args.reference not in dataset.videos_info or args.driving not in dataset.videos_info:
        p.error("Reference and driving IDs must exist in the dataset index")
    source = dataset._load_source_info(args.reference)
    target = dataset._load_target_info(args.driving)
    target["source_info"] = source["source_info"]
    target["source_image"] = source["source_image"]
    target["head_image"] = source["head_image"]
    target["prompt"] = dataset.videos_promt[args.driving]
    pipe = WanVideoPipeline.from_pretrained(
        config=WAN_CONFIGS["animate-14B"],
        checkpoint_dir=args.model_path,
        assets=args.assets,
        device="cuda",
        offload_device="cpu",
        torch_dtype=torch.bfloat16,
        convert_model_dtype=True,
    )
    target["target_infos"] = retarget_identity(
        source["source_info"], target["target_infos"], pipe.ehm
    )
    pipe.load_lora_extra(args.checkpoint)
    pipe.enable_vram_management()
    frames, _ = pipe(
        data=target,
        prompt=target["prompt"],
        clip_len=args.num_frames,
        seed=args.seed,
        num_inference_steps=args.steps,
        add_refer=args.spatial_anchor,
        sample_solver=args.sample_solver,
    )
    if args.width is None and args.resolution_area is None:
        record = dataset.videos_info[args.reference]
        width, height = record.get("width", dataset.width), record.get("height", dataset.height)
        left, top = (dataset.width - width) // 2, (dataset.height - height) // 2
        frames = [np.asarray(frame)[top : top + height, left : left + width] for frame in frames]
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fps = (
        args.fps
        or dataset.videos_info[args.driving].get("fps")
        or WAN_CONFIGS["animate-14B"].sample_fps
    )
    height, width = np.asarray(frames[0]).shape[:2]
    pixel_format = "yuv444p" if width % 2 or height % 2 else "yuv420p"
    save_video(frames, args.output, fps=fps, quality=8, pixelformat=pixel_format)
    print(f"Saved {len(frames)} frames.")


if __name__ == "__main__":
    main()
