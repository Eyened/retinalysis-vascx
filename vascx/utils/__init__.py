import numpy as np

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
