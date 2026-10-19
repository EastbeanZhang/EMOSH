import random

import numpy as np

from models.utils.augmentation import apply_transform


def test_reference_and_control_keep_the_same_geometry():
    random.seed(7)
    image = np.zeros((96, 128, 3), dtype=np.float32)
    image[20:75, 35:95] = 0.8
    transformed, mask, control = apply_transform(image, image * 255)

    assert transformed.shape == image.shape == control.shape
    assert mask.shape == (96, 128, 1)
    assert set(np.unique(mask)) <= {0, 1}
    assert 0 < mask.sum() <= 96 * 128
    assert transformed.max() > 0
    np.testing.assert_allclose(transformed * 255, control, atol=1e-4)
