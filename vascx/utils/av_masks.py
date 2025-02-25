import numpy as np


def onehot_to_rgb(segmentation):
    """
    Convert a grayscale segmentation map to an RGB image.

    Parameters:
        segmentation (numpy.ndarray): Grayscale image with labels
                                      0 (background),
                                      1 (arteries),
                                      2 (veins),
                                      3 (overlap).

    Returns:
        numpy.ndarray: RGB image where:
                       - Arteries are in the red channel.
                       - Veins are in the blue channel.
                       - Overlaps are in both red and blue.
    """
    # Ensure input is a NumPy array
    segmentation = np.asarray(segmentation)

    # Create an empty RGB image
    height, width = segmentation.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign channels
    rgb_image[..., 0] = (segmentation == 1) | (segmentation == 3)  # Red for arteries
    rgb_image[..., 2] = (segmentation == 2) | (segmentation == 3)  # Blue for veins

    # Scale to 255 for visualization
    rgb_image *= 255

    return rgb_image
