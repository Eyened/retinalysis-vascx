from warnings import warn

import numpy as np
from rtnls_enface.base import Point
from scipy import ndimage as ndi
from skimage import measure


def remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """Remove objects smaller than the specified size.

    Expects ar to be an array with labeled objects, and removes objects
    smaller than min_size. If `ar` is bool, the image is first labeled.
    This leads to potentially different behavior for bool and 0-and-1
    arrays.

    Parameters
    ----------
    ar : ndarray (arbitrary shape, int or bool type)
        The array containing the objects of interest. If the array type is
        int, the ints must be non-negative.
    min_size : int, optional (default: 64)
        The smallest allowable object size.
    connectivity : int, {1, 2, ..., ar.ndim}, optional (default: 1)
        The connectivity defining the neighborhood of a pixel. Used during
        labelling if `ar` is bool.
    out : ndarray
        Array of the same shape as `ar`, into which the output is
        placed. By default, a new array is created.

    Raises
    ------
    TypeError
        If the input array is of an invalid type, such as float or string.
    ValueError
        If the input array contains negative values.

    Returns
    -------
    out : ndarray, same shape and type as input `ar`
        The input array with small connected components removed.

    Examples
    --------
    >>> from skimage import morphology
    >>> a = np.array([[0, 0, 0, 1, 0],
    ...               [1, 1, 1, 0, 0],
    ...               [1, 1, 1, 0, 1]], bool)
    >>> b = morphology.remove_small_objects(a, 6)
    >>> b
    array([[False, False, False, False, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True, False, False]])
    >>> c = morphology.remove_small_objects(a, 7, connectivity=2)
    >>> c
    array([[False, False, False,  True, False],
        [ True,  True,  True, False, False],
        [ True,  True,  True, False, False]])
    >>> d = morphology.remove_small_objects(a, 6, out=a)
    >>> d is a
    True

    """
    if out is None:
        out = ar.copy()
    else:
        out[:] = ar

    if min_size == 0:  # shortcut for efficiency
        return out

    if out.dtype == bool:
        footprint = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs = np.zeros_like(ar, dtype=np.int32)
        ndi.label(ar, footprint, output=ccs)
    else:
        ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError(
            "Negative value labels are not supported. Try "
            "relabeling the input with `scipy.ndimage.label` or "
            "`skimage.morphology.label`."
        )

    if len(component_sizes) == 2 and out.dtype != bool:
        warn(
            "Only one label was provided to `remove_small_objects`. "
            "Did you mean to use a boolean array?"
        )

    too_small = component_sizes < min_size
    too_small_mask = too_small[ccs]
    out[too_small_mask] = 0

    return out


def fill_small_holes(mask, area_threshold=25):
    inverted_image = np.invert(mask)
    labeled_image, num_features = measure.label(
        inverted_image, return_num=True, connectivity=1, background=0
    )
    properties = measure.regionprops(labeled_image)
    for prop in properties:
        if prop.area < area_threshold:
            labeled_image[labeled_image == prop.label] = 0
    # Invert the image back to original form where small holes are now filled
    return np.invert(labeled_image > 0)


def binarize_and_fill(mask, threshold=0.5, area_threshold=25):
    # binarize
    bin = np.empty(mask.shape, dtype=np.uint8)
    bin[mask < threshold] = 0
    bin[mask >= threshold] = 255

    # fill in small holes in the segmentation mask
    filled = fill_small_holes(bin, area_threshold=area_threshold)
    return filled


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
