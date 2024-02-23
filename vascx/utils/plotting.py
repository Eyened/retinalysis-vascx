from math import ceil
from random import randint
from typing import List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_disc(image, mask, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
        ax.set_axis_off()

    ax.imshow(image.squeeze())
    masked = np.ma.masked_where(mask == 0, mask)
    ax.imshow(masked, alpha=0.5)


def plot_discs(pairs, ncols=4):
    nrows = len(pairs) // ncols

    def plot_ax(i, ax):
        plot_disc(pairs[i][0], pairs[i][1], ax)

    plot_grid(plot_ax, nrows, ncols)


def plot_grid(plot_fn, nrows=4, ncols=4):
    fig = plt.figure(figsize=(16, 16))
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    for i, ax in enumerate(axs):
        plot_fn(i, ax)


def plot_gridfns(plot_fns: List[None], ncols=4):
    nrows = ceil(len(plot_fns) / ncols)
    fig = plt.figure(figsize=(16, 20))
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    for i, fn in enumerate(plot_fns):
        fn(ax=axs[i], fig=fig)


def plot_image_grid(images):
    nrows = len(images)
    ncols = len(images[0])

    def plot_fn(i, ax):
        row = i // ncols
        col = i % ncols
        ax.imshow(images[row][col])

    return plot_grid(plot_fn, nrows, ncols)


def plot_model_outputs(model, dataset, nrows=4, ncols=4, first=None):
    N = nrows * ncols
    if first is None:
        first = randint(0, len(dataset) - N)
    last = first + N

    batch_test = [dataset[i]["image"][np.newaxis, :, :, :] for i in range(first, last)]
    batch_orig = [
        dataset.getitem(i)["image"].permute(1, 2, 0) for i in range(first, last)
    ]

    # fig, axs = plt.subplots(nrows, ncols, figsize=(16,16))
    fig = plt.figure(figsize=(16, 16))
    axs = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(nrows, ncols),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )
    for im_test, im_orig, ax in zip(batch_test, batch_orig, axs):
        mask = model(im_test).detach().numpy().squeeze()
        masked = np.ma.masked_where(mask == 0, mask)
        ax.imshow(im_orig.numpy().squeeze())
        ax.imshow(masked, alpha=0.5)


def blend_images(img1, img2, mask):
    """
    Blend two images based on a mask.

    Parameters:
    - img1: First image (numpy array).
    - img2: Second image (numpy array).
    - mask: Binary mask (numpy array) where True indicates pixels from img1 and False indicates pixels from img2.

    Returns:
    - Blended image.
    """

    if img1.shape != img2.shape or img1.shape[:2] != mask.shape:
        raise ValueError("The dimensions of the images and the mask must match.")

    # Use mask to select pixels from img1
    blended_img1 = img1 * mask[..., np.newaxis].astype(img1.dtype)

    # Use inverted mask to select pixels from img2
    blended_img2 = img2 * (1 - mask[..., np.newaxis].astype(img2.dtype))

    # Combine the images to get the blended result
    blended_result = blended_img1 + blended_img2

    return blended_result


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


def resize_image_by_height(image, new_width):
    """
    Resize an image to a specific width while maintaining its aspect ratio.

    :param image: The input image as a NumPy array.
    :param new_width: The desired new width of the image.
    :return: The resized image.
    """
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the new height to maintain the aspect ratio
    aspect_ratio = original_height / original_width
    new_height = int(new_width * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    return resized_image


resize_image_by_width = resize_image_by_height
