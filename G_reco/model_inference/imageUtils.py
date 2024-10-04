import numpy as np
from PIL import Image


def saveImage(array, path):
    im = Image.fromarray(array)
    im.save(path)

def readImageConvertGray(image_path):
    im = Image.open(image_path).convert('LA')
    im = np.array(im)
    im = im[:im.shape[0], :im.shape[1], 0]
    return im

def readImage(image_path):
    im = Image.open(image_path)
    im = np.array(im)
    return im

def normalizeImage(image, max_value):
    min = np.amin(image)
    max = np.amax(image)
    n_im = ((image.astype(np.float64) - min)/(max-min))*max_value
    return n_im










