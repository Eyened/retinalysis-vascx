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
