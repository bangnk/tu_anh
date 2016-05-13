from __future__ import unicode_literals

import numpy as np
from PIL import Image
import os
import json
import cv2
import cPickle
from wrappers import *
from tiny import n_clusters


class HistModel:
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/kmean_model.pkl', 'r') as f:
        model = cPickle.load(f)
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/sift_hist.npy', 'r') as f:
        histograms = cPickle.load(f)
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/sift_names.npy', 'r') as f:
        names = cPickle.load(f)

    def sdd(self, file):
        img = Image.open(file).convert('L')
        img.thumbnail((32, 32))
        img_hist = img.histogram()
        hists = np.load(os.path.dirname(os.path.realpath(__file__)) + '/hists.npy')
        s = np.sum((hists - img_hist) ** 2, axis=1)
        s = s.argsort()[:15]
        with open('hists/names.json', 'r') as f:
            names = json.load(f)
        s = [names[str(i)] for i in s]
        return s

    @elapsed()
    def cosine_sift(self, uploaded_file):
        sift = cv2.xfeatures2d.SIFT_create()
        nparr = np.fromstring(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        kp, des = sift.detectAndCompute(img, None)

        v = HistModel.model.predict(des)
        histogram = np.histogram(v, bins=n_clusters, range=(0, n_clusters))[0]

        s = np.dot(HistModel.histograms, histogram)
        indexes = np.array(s).argsort()[::-1][:15]
        nam = [HistModel.names[i] for i in indexes]
        return nam