import glob
import os
import shutil
from pathlib import Path
from typing import List, Union

import numpy as np
import pydicom
from PIL import Image

from rtnls_enface.utils.data_loading import open_mask


def load_av_segmentation(fpath, threshold=0.5):
    im = open_mask(fpath)

    if len(im.shape) == 2:
        # single channel image
        red = np.zeros((im.shape[0], im.shape[1]))
        red[(im == 1)] = 1

        blue = np.zeros((im.shape[0], im.shape[1]))
        blue[(im == 2)] = 1

        green = np.zeros((im.shape[0], im.shape[1]))
        green[(im == 3)] = 1

        return {
            "arteries": np.logical_or(red, green),
            "veins": np.logical_or(blue, green),
        }

    else:
        # three-channel image
        if len(np.unique(im)) > 2:
            t = round(threshold * np.max(im))
            bin = np.empty(im.shape)
            bin[im < t] = 0
            bin[im >= t] = 1
            im = bin

        unique_vals = np.unique(im)
        assert len(unique_vals) <= 2, f"found unique {np.unique(im)}"
        white_value = unique_vals[1] if len(unique_vals) > 1 else 1.0

        red = np.zeros((im.shape[0], im.shape[1]))
        red[(im[:, :, 0] == white_value)] = 1

        green = np.zeros((im.shape[0], im.shape[1]))
        green[(im[:, :, 1] == white_value)] = 1

        blue = np.zeros((im.shape[0], im.shape[1]))
        blue[(im[:, :, 2] == white_value)] = 1

        return {
            "arteries": np.logical_or(red, green),
            "veins": np.logical_or(blue, green),
        }


def load_image_pil(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    if path.suffix == ".dcm":
        ds = pydicom.dcmread(str(path))
        img = Image.fromarray(ds.pixel_array)
    else:
        img = Image.open(str(path))
    return img


def load_image(path: Union[Path, str], dtype: Union[np.uint8, np.float32] = np.uint8):
    if Path(path).suffix == ".npy":
        im = np.load(path)
    else:
        im = np.array(load_image_pil(path), dtype=np.uint8)
    if im.dtype == np.uint8 and dtype == np.float32:
        im = (im / 255).astype(np.float32)
    if im.dtype == np.float32 and dtype == np.uint8:
        im = np.round(im * 255).astype(np.uint8)
    return im


def linear_interpolate(image, x, y):
    """
    Perform linear interpolation for a single-channel (grayscale) image at a given point (x, y).

    Parameters:
        image (np.ndarray): Grayscale image as a 2D NumPy array.
        x (float): X-coordinate (horizontal) of the point.
        y (float): Y-coordinate (vertical) of the point.

    Returns:
        float: Interpolated intensity value at (x, y).
    """
    height, width = image.shape

    # Ensure the coordinates are within bounds
    if x < 0 or x >= width - 1 or y < 0 or y >= height - 1:
        raise ValueError("Point is out of bounds for linear interpolation.")

    # Find the integer coordinates around (x, y)
    x0, y0 = int(x), int(y)
    x1, y1 = x0 + 1, y0 + 1

    # Get the intensity values at the four surrounding pixels
    I00 = image[y0, x0]
    I10 = image[y0, x1]
    I01 = image[y1, x0]
    I11 = image[y1, x1]

    # Compute weights for linear interpolation
    wx, wy = x - x0, y - y0

    # Interpolate in the x-direction first, then in the y-direction
    I0 = (1 - wx) * I00 + wx * I10  # Interpolate between (x0, y0) and (x1, y0)
    I1 = (1 - wx) * I01 + wx * I11  # Interpolate between (x0, y1) and (x1, y1)
    I = (1 - wy) * I0 + wy * I1  # Interpolate between I0 and I1 in the y-direction

    return I
