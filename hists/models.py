from __future__ import unicode_literals

import numpy as np
from PIL import Image
import os
import json


class HistModel:

    def sdd(self, file):
        img = Image.open(file).convert('L')
        img.thumbnail((32, 32))
        img_hist = img.histogram()
        hists = np.load(os.path.dirname(os.path.realpath(__file__)) + '/hists.npy')
        s = np.sum((hists - img_hist) ** 2, axis=1)
        s = s.argsort()[:10]
        with open('hists/names.json', 'r') as f:
            names = json.load(f)
        s = [names[str(i)] for i in s]
        return s