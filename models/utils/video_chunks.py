import torch


def padded_length(frame_count, clip_length, overlap):
    if frame_count < 1 or clip_length <= overlap:
        raise ValueError("A video needs frames and a clip length larger than its overlap.")
    stride = clip_length - overlap
    extra = (-(frame_count - overlap)) % stride
    return max(clip_length, frame_count + extra)


def temporal_reference(chunks, count):
    remaining = count
    tail = []
    for chunk in reversed(chunks):
        take = min(remaining, chunk.shape[2])
        tail.append(chunk[:, :, -take:])
        remaining -= take
        if remaining == 0:
            break
    if remaining:
        raise ValueError("Not enough generated frames for temporal conditioning.")
    return torch.cat(tail[::-1], dim=2)[0].clone().detach()
