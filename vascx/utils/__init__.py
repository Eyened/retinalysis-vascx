def bounding_box(points):
    """
    Returns the tightest bounding box for a list of 2D points.

    Args:
    - points (list of tuple): List of 2D points where each point is represented as (x, y)

    Returns:
    - (tuple): A tuple containing two tuples. The first tuple represents the bottom-left corner of the bounding box
               and the second tuple represents the top-right corner of the bounding box.
    """

    min_x, min_y = float("inf"), float("inf")
    max_x, max_y = float("-inf"), float("-inf")

    for x, y in points:
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return ((min_x, min_y), (max_x, max_y))


def pad_bounding_box(corner1, corner2, padding):
    """
    Pads a bounding box by a specified number of pixels.

    Args:
    - corner1 (tuple): Top-left corner coordinates as (x1, y1).
    - corner2 (tuple): Bottom-right corner coordinates as (x2, y2).
    - padding (int): Number of pixels to pad the bounding box.

    Returns:
    - (tuple): A tuple containing two tuples. The first tuple represents the new top-left corner and the second
               tuple represents the new bottom-right corner of the padded bounding box.
    """
    x1, y1 = corner1
    x2, y2 = corner2

    return ((x1 - padding, y1 - padding), (x2 + padding, y2 + padding))


def concat(*args):
    """
    Concatenates an arbitrary number of lists.

    Args:
    - *args (multiple lists): An arbitrary number of lists to be concatenated.

    Returns:
    - list: A concatenated list.
    """
    result = []
    for lst in args:
        result.extend(lst)
    return result
