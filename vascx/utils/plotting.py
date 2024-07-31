def find_bounding_box(points, padding=10):
    """
    Find the bounding box for a list of points.

    :param points: A list of points, where each point is represented as a tuple (x, y).
    :return: A tuple of two points representing the bounding box: (bottom_left, top_right).
    """
    if len(points) <= 1:
        return None

    min_x = min(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_x = max(points, key=lambda p: p[0])[0]
    max_y = max(points, key=lambda p: p[1])[1]

    top_left = (min_x, min_y)
    bottom_right = (max_x, max_y)

    return pad_bounding_box(top_left, bottom_right, padding)


def pad_bounding_box(top_left, bottom_right, padding):
    """
    Pad a bounding box on all sides by a fixed amount.

    :param top_left: A tuple (x, y) representing the top left corner of the bounding box.
    :param bottom_right: A tuple (x, y) representing the bottom right corner of the bounding box.
    :param padding: The amount by which to pad each side of the bounding box.
    :return: A tuple containing the new top left and bottom right corners of the padded bounding box.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Pad the bounding box
    new_top_left = (x1 - padding, y1 - padding)
    new_bottom_right = (x2 + padding, y2 + padding)

    return new_top_left, new_bottom_right
