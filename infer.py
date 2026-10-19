import argparse

import torch

from models.utils.data_utils import Tracked_VideoDataset_infer
from models.utils.general_utils import save_video
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
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=77)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disable-spatial-anchor", action="store_true")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires an NVIDIA CUDA GPU.")
    dataset = Tracked_VideoDataset_infer(
        base_path=args.dataset,
        split="test",
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    if args.reference not in dataset.videos_info or args.driving not in dataset.videos_info:
        p.error("Reference and driving IDs must exist in videos_info.json")
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
        add_refer=not args.disable_spatial_anchor,
    )
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, args.output, fps=args.fps, quality=8)
    print(f"Saved {len(frames)} frames.")


if __name__ == "__main__":
    main()
