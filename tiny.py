import glob
import os
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import json


def tiny():
    glo = glob.glob('shopping/images/*.jpg')
    size = 32, 32
    for i, filename in enumerate(glo):
        if i % 100 == 0 and i > 0:
            print i
        filename_details = filename.split("/")
        name, ext = os.path.splitext(filename_details[-1])
        im = Image.open(filename).convert('L')
        im.thumbnail(size)
        im.save('shopping/grayscale32x32/' + name + "_32x32.png", "PNG")


def hist():
    glo = glob.glob('shopping/grayscale32x32/*.png')
    hists = np.zeros((len(os.listdir('shopping/grayscale32x32')), 256))
    names = {}
    for i, filename in enumerate(glo):
        if i % 100 == 0 and i > 0:
            print i
        im = Image.open(filename)
        hists[i] = im.histogram()
        filename_details = filename.split("/")
        names[int(i)] = filename_details[-1].replace('_32x32.png', '.jpg')
    np.save('hists/hists.npy', hists)
    with open('hists/names.json', 'w') as f:
        json.dump(names, f)


if __name__ == "__main__":
    if len(os.listdir('shopping/grayscale32x32')) == 0:
        tiny()
    if not os.path.exists('hists/hists.npy'):
        hist()