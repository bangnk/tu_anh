from __future__ import unicode_literals

import numpy as np
from PIL import Image
import os
import json
import cv2
import cPickle


class HistModel:
    def cosine_measure(self, v1, v2):
        prod = np.dot(v1, v2)
        len1 = np.sqrt(np.sum(v1 ** 2))
        len2 = np.sqrt(np.sum(v2 ** 2))
        return prod / (len1 * len2)

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

    def cosine_sift(self, file):
        sift = cv2.xfeatures2d.SIFT_create()
        nparr = np.fromstring(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        kp, des = sift.detectAndCompute(img, None)

        with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/kmean_model.pkl', 'r') as f:
            model = cPickle.load(f)
        with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/sift_tfidf.npy', 'r') as f:
            histograms = cPickle.load(f)

        v = model.predict(des)
        histogram = np.zeros(100)
        for j in v:
            histogram[j] += 1
        print histogram
        print self.cosine_measure(histogram, histogram)
        s = []
        with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/sift_names.npy', 'r') as f:
            names = cPickle.load(f)
        for h in range(len(histograms)):
            # print histograms[h, :]
            # print
            cos = self.cosine_measure(histogram, histograms[h, :])
            if np.sum(histograms[h, :] == histogram) == 100:
                print names[h], len(names)
            s.append(cos)
        s = np.array(s)
        indexes = np.array(s).argsort()[::-1][:15]
        nam = [names[i] for i in indexes]
        return nam