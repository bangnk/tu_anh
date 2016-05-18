from __future__ import unicode_literals

import numpy as np
from PIL import Image
import os
import json
import cv2
import cPickle
from wrappers import *
from tiny import n_clusters
from sklearn import preprocessing
import scipy.sparse


class SearchModel:
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/kmean_model.pkl', 'r') as f:
        model = cPickle.load(f)
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/lshforest_combine.pkl', 'r') as f:
        lsh = cPickle.load(f)
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/vectorizer.pkl', 'r') as f:
        vectorizer = cPickle.load(f)
    with open(os.path.dirname(os.path.realpath(__file__)) + '/../data/sift_names.pkl', 'r') as f:
        names = cPickle.load(f)

    def sdd(self, uploaded_file):
        """
        Find similar image base on SDD of histogram
        """
        img = Image.open(uploaded_file).convert('L')
        img.thumbnail((32, 32))
        img_hist = img.histogram()
        hists = np.load(os.path.dirname(os.path.realpath(__file__)) + '/hists.pkl')
        s = np.sum((hists - img_hist) ** 2, axis=1)
        s = s.argsort()[:15]
        with open('hists/names.json', 'r') as f:
            names = json.load(f)
        s = [names[str(i)] for i in s]
        return s

    @elapsed()
    def cosine_sift(self, uploaded_file):
        """
        Find similar image base on sift features
        """
        name = uploaded_file.name
        name = name.replace('img', 'descr')
        name1 = name.replace('.jpg', '.txt')

        name = os.path.dirname(os.path.realpath(__file__)) + '/../shopping/queryimages/' + name1
        if not os.path.exists(name):
            name = os.path.dirname(os.path.realpath(__file__)) + '/../shopping/images/' + name1
        filenames = [name]
        text_hist = SearchModel.vectorizer.transform(filenames).tocsr()
        preprocessing.normalize(text_hist, copy=False)

        sift = cv2.xfeatures2d.SIFT_create()
        nparr = np.fromstring(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        kp, des = sift.detectAndCompute(img, None)

        v = SearchModel.model.predict(des)
        sift_hist = np.histogram(v, bins=n_clusters, range=(0, n_clusters))[0]
        sift_hist = np.reshape(sift_hist, (1, len(sift_hist)))

        lamb = .5
        histogram = scipy.sparse.hstack([text_hist * lamb, sift_hist * (1-lamb)]).toarray()
        preprocessing.normalize(histogram, copy=False)

        indices = SearchModel.lsh.kneighbors(histogram, n_neighbors=24)[1][0]
        names = [SearchModel.names[i] for i in indices]
        return names