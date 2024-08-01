import numpy as np
from skimage import measure

from rtnls_enface.base import Point


def binarize_and_fill(mask, threshold=0.5, area_threshold=25):
    # binarize
    bin = np.empty(mask.shape, dtype=np.uint8)
    bin[mask < threshold] = 0
    bin[mask >= threshold] = 255

    # fill in small holes in the segmentation mask
    inverted_image = np.invert(bin)
    labeled_image, num_features = measure.label(
        inverted_image, return_num=True, connectivity=1, background=0
    )
    properties = measure.regionprops(labeled_image)
    for prop in properties:
        if prop.area < area_threshold:
            labeled_image[labeled_image == prop.label] = 0

    # Invert the image back to original form where small holes are now filled
    return np.invert(labeled_image > 0)


def min_distance_to_edge(image: np.array, point: Point):
    # Image dimensions
    height, width = image.shape[:2]

    # Point coordinates
    x, y = point.x, point.y

    # Distances to each edge
    distance_to_left_edge = x
    distance_to_right_edge = width - 1 - x
    distance_to_top_edge = y
    distance_to_bottom_edge = height - 1 - y

    # Minimum distance to any edge
    return min(
        distance_to_left_edge,
        distance_to_right_edge,
        distance_to_top_edge,
        distance_to_bottom_edge,
    )
