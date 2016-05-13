import glob
import os
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
import cPickle
from sklearn.cluster import MiniBatchKMeans
from wrappers import *
from collections import Counter
np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing

n_clusters = 100
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


def sift_features():
    glo = glob.glob('shopping/images/*.jpg')
    sift = cv2.xfeatures2d.SIFT_create()
    dess = []
    filenames = []
    n_des = []
    total_des = 0
    for i, filename in enumerate(glo):
        if i % 100 == 0 and i > 0:
            print i
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        filenames.append(filename)
        dess.append(des)
        n_des.append(len(des))
        total_des += len(des)
    df = pd.DataFrame()
    df['n_des'] = n_des
    df['filename'] = filenames
    df.to_csv('data/sift_info.csv', index=False)
    des_mat = np.vstack(dess)  # Features size: (664075, 128)
    print 'Features size: %s' % str(des_mat.shape)

    with open('data/sift_features.npy', 'w') as f:
        cPickle.dump(des_mat, f)


@elapsed()
def sift_kmean():
    with open('data/sift_features.npy', 'r') as f:
        xtrain = cPickle.load(f)
    # preprocessing.normalize(xtrain, axis=0, copy=False)
    model = MiniBatchKMeans(n_clusters=n_clusters, verbose=1, batch_size=1000, max_no_improvement=100, init_size=10000)
    model.fit(xtrain)
    z = model.predict(xtrain)
    with open('data/sift_leaders.npy', 'w') as f:
        cPickle.dump(z, f)
    with open('data/kmean_model.pkl', 'w') as f:
        cPickle.dump(model, f)


def sift_hist():
    with open('data/sift_leaders.npy', 'r') as f:
        clusters = cPickle.load(f)
    images = pd.read_csv('data/sift_info.csv')
    histograms = np.zeros((len(images), n_clusters))
    t = 0
    names = []
    for i in range(len(images)):
        image = images.iloc[i]
        for j in range(image.n_des):
            histograms[i, clusters[t]] += 1
            t += 1
        names.append(image.filename.split('/')[-1])
        if 'bags_hobo_289' in image.filename:
            print histograms[i]
            h = [  0,   0,   3,   0,   2,   2,   0,   0,   0,   2,   1,   1,   1,   0,   3,
                   3,   0,  10,   1,   0,   5,   0,   3,   0,   1,   1,   1,   1,   1,   0,
                   6,   1,   1,   1,   1,   3,   0,   3,   7,   2,   8,   0,   0,   4,   1,
                   0,   0,   2,   0,   1,   1,   2,   0,   0,   0,   3,   0,   1,   5,   1,
                   1,   1,   1,   0,   2,   1,   0,   0,   0,   0,   4,   2,   1,   0,   1,
                   0,   2,   0,   3,   3,   2,   0,   0,   0,   0,   0,   1,   0,   1,   0,
                   0,   0,   2,   0,   2,   0,   0,   0,   0,   3]
            print np.sum(h == histograms[i])
    # df = np.sum(preprocessing.binarize(histograms, copy=False), axis=0)

    # print tfidf.shape
    # histograms = preprocessing.normalize(histograms)
    # print histograms
    with open('data/sift_names.npy', 'w') as f:
        cPickle.dump(names, f)
    with open('data/sift_tfidf.npy', 'w') as f:
        cPickle.dump(histograms, f)


if __name__ == "__main__":
    if len(os.listdir('shopping/grayscale32x32')) == 0:
        tiny()
    if not os.path.exists('hists/hists.npy'):
        hist()
    if not os.path.exists('data/sift_features.npy'):
        sift_features()
    if not os.path.exists('data/sift_leaders.npy'):
        sift_kmean()
    # if not os.path.exists('data/sift_tfidf.npy'):
    sift_hist()

