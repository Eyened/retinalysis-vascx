import cv2
import numpy as np


def undo_crop(cropped, original_shape, padding, crop_position):
    assert len(original_shape) >= 2, "Invalid original_shape"

    channel_dim = cropped.shape[2] if len(cropped.shape) > 2 else 1
    result = np.zeros(
        (original_shape[0], original_shape[1], channel_dim), dtype=cropped.dtype
    ).squeeze()

    min_x, min_y, max_x, max_y = crop_position
    # Place the cropped image onto the canvas
    crop_w, crop_h = max_x - min_x, max_y - min_y

    top, bottom, left, right = padding

    unpadded = cropped[
        top : (-bottom if bottom > 0 else None), left : (-right if right > 0 else None)
    ]
    patch = cv2.resize(unpadded, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    result[min_y:max_y, min_x:max_x, ...] = patch

    return result


def undo_crop_keypoint(point, crop_position, padding):
    """
    Maps a point from the cropped image back to the original image.

    Parameters:
    - point: Tuple[int, int], the point (x, y) in the cropped image.
    - crop_position: Tuple[int, int, int, int], the bounding box (min_x, min_y, max_x, max_y) of the crop in the original image.
    - padding: Tuple[int, int, int, int], the padding (top, bottom, left, right) applied to the cropped image.
    - original_shape: Tuple[int, int], the shape (height, width) of the original image.

    Returns:
    - Tuple[int, int], the point (x, y) in the original image.
    """
    # Unpack the parameters
    x, y = point
    min_x, min_y, max_x, max_y = crop_position
    crop_w, crop_h = (max_x - min_x), (max_y - min_y)

    top, bottom, left, right = padding
    x, y = x - left, y - top
    x = x * max(crop_w, crop_h) / 1024
    y = y * max(crop_w, crop_h) / 1024

    x += min_x
    y += min_y

    return x, y
