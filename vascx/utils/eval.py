import numpy as np


def dice_score(image1, image2):
    intersection = np.sum(image1 * image2)
    total = np.sum(image1) + np.sum(image2)
    if total == 0:
        return 1.0
    dice = 2 * intersection / total
    return dice
