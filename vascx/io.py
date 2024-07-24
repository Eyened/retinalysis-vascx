from abc import ABC,abstractmethod
from typing import Dict

from PIL import Image
import numpy as np

def open_image(fpath):
    im = Image.open(fpath)

    # if RGBA, paste into black background before converting to 'L'
    if im.mode == 'RGBA':
        new_image = Image.new("RGBA", im.size, "BLACK")
        new_image.paste(im, (0, 0), im)
        im = new_image.convert('RGB')

    if im.mode != 'RGB':
        im = im.convert('RGB')

    return np.array(im)
    
class SegmentationReader(ABC):

    @abstractmethod
    def read(self, fpath: str) -> Dict[str, np.ndarray]:
        pass


class BasicReader(SegmentationReader):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def read(self, fpath: str):
        im = open_image(fpath)

        if(len(np.unique(im)) > 2):
            t = round(self.threshold * np.max(im))
            bin = np.empty(im.shape)
            bin[im < t] = 0
            bin[im >= t] = 1
            im = bin

        assert len(np.unique(im)) <= 2, f'found unique {np.unique(im)}'
        white_value = np.unique(im)[1]

        # all channels are equal -> vessels image
        if ((im[:, :, 0] == im[:, :, 1]).all() and (im[:, :, 1] == im[:, :, 2]).all()):
            vessels = np.zeros((im.shape[0], im.shape[1]))
            
            vessels[(im[:, :, 0] == white_value)] = 1
            layers = {'vessels': vessels}
        else:
            # layers are not equal -> artery-vein image
            red = np.zeros((im.shape[0], im.shape[1]))
            red[(im[:, :, 0] == white_value) & (im[:, :, 1] == 0)
                & (im[:, :, 2] == 0)] = 1
            
            green = np.zeros((im.shape[0], im.shape[1]))
            green[(im[:, :, 1] == white_value) & (im[:, :, 0] == 0)
                    & (im[:, :, 2] == 0)] = 1
            
            blue = np.zeros((im.shape[0], im.shape[1]))
            blue[(im[:, :, 2] == white_value) & (im[:, :, 0] == 0)
                    & (im[:, :, 1] == 0)] = 1
            
            layers = {
                'arteries': np.logical_or(red, green),
                'veins': np.logical_or(blue, green)
            }

        return layers
    

class NonOverlappingReader(SegmentationReader):
    '''
    '''
    def __init__(self, layers = None):
        super().__init__()
        if layers is None:
            layers =  {
                "arteries": [1,3],
                "veins": [2,3]
            }
        self.layers = layers

    def read(self, fpath: str):
        im = Image.open(fpath)
        im = np.array(im)

        layers = {}
        for name, labels in self.layers.items():
            layers[name] = (np.isin(im, labels))

        return layers
