import albumentations as A
import cv2
import numpy as np


def create_transform(h, w):
    return A.Compose(
        [
            A.Affine(
                scale=(0.7, 1.3),
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-30, 30),
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                p=1.0,
            ),
            A.PadIfNeeded(
                min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
            ),
            A.RandomCrop(height=h, width=w, p=1.0),
        ],
        additional_targets={"mesh_image": "image"},
    )


def apply_transform(image, mesh_image=None):
    transform = create_transform(image.shape[0], image.shape[1])
    mask = np.ones((image.shape[0], image.shape[1], 1), dtype=np.float32)
    if mesh_image is not None:
        augmented = transform(image=image, mask=mask, mesh_image=mesh_image)
        transformed_mesh_image = augmented["mesh_image"]
    else:
        augmented = transform(image=image, mask=mask)
        transformed_mesh_image = None
    transformed_image = augmented["image"]
    transformed_mask = augmented["mask"]
    return transformed_image, transformed_mask, transformed_mesh_image
