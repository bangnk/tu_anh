import glob
import os
import cv2
import numpy as np
from PIL import Image
import json
import pandas as pd
import cPickle
from sklearn.cluster import MiniBatchKMeans
np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing
from sklearn.neighbors import LSHForest
n_clusters = 5000
from wrappers import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold
from sklearn.utils import shuffle
import scipy.sparse
from sklearn.metrics import precision_score
from collections import Counter


def tiny():
    """
    Shrink images and convert to grayscale
    """
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
    """
    Calculate image histograms on grayscale
    """
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
    np.save('hists/hists', hists)
    with open('hists/names.json', 'w') as f:
        json.dump(names, f)


def sift_features():
    """
    Extract sift features
    """
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

    with open('data/sift_features.pkl', 'w') as f:
        cPickle.dump(des_mat, f)


@elapsed()
def sift_kmean():
    """
    Use kmean to get visual words
    """
    with open('data/sift_features.pkl', 'r') as f:
        xtrain = cPickle.load(f)
    model = MiniBatchKMeans(n_clusters=n_clusters, verbose=0, batch_size=1000, max_no_improvement=100, init_size=10000)
    model.fit(xtrain)
    z = model.predict(xtrain)
    with open('data/sift_leaders.pkl', 'w') as f:
        cPickle.dump(z, f)
    with open('data/kmean_model.pkl', 'w') as f:
        cPickle.dump(model, f)


def sift_hist():
    """
    Calculate histogram of visual words and make index for fast search (lshmodel)
    """
    with open('data/sift_leaders.pkl', 'r') as f:
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

    histograms = preprocessing.normalize(histograms)
    with open('data/sift_hist.pkl', 'w') as f:
        cPickle.dump(histograms, f)
    with open('data/sift_names.pkl', 'w') as f:
        cPickle.dump(names, f)


def text_hist():
    """
    Calculate histogram of text of images
    """
    with open('data/sift_names.pkl', 'r') as f:
        names = cPickle.load(f)
    with open('data/sift_hist.pkl', 'r') as f:
        sift_hists = cPickle.load(f)
    filenames = []
    for name in names:
        name = name.replace('img', 'descr')
        name = name.replace('.jpg', '.txt')
        filenames.append('shopping/images/' + name)
    vectorizer = CountVectorizer(input='filename', token_pattern="(?u)"+'\w+', ngram_range=(1, 1), min_df=2)
    xall_transformed = vectorizer.fit_transform(filenames).tocsr()
    preprocessing.normalize(xall_transformed, copy=False)

    lamb = .6
    hists = scipy.sparse.hstack([xall_transformed * lamb, sift_hists * (1-lamb)]).toarray()
    preprocessing.normalize(hists, copy=False)
    model = LSHForest()
    model.fit(hists)
    with open('data/text_hist.pkl', 'w') as f:
        cPickle.dump(xall_transformed, f)
    with open('data/vectorizer.pkl', 'w') as f:
        cPickle.dump(vectorizer, f)
    with open('data/lshforest_combine.pkl', 'w') as f:
        cPickle.dump(model, f)


def tune_cv():
    """
    cross validate to find weight for text and sift
    """
    with open('data/text_hist.pkl', 'r') as f:
        text_hists = cPickle.load(f)
    with open('data/sift_hist.pkl', 'r') as f:
        sift_hists = cPickle.load(f)
    with open('data/sift_names.pkl', 'r') as f:
        names = cPickle.load(f)

    kf = KFold(len(names), n_folds=5)
    indexes = []
    for train_index, valid_index in kf:
        indexes += [[train_index, valid_index]]

    y = []
    for n in names:
        a = n.split('.')[0].split('_')
        y.append(a[1] + a[2])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    y, text_hists, sift_hists = shuffle(y, text_hists, sift_hists, random_state=123456)

    lambs = np.array([i for i in range(11)]) / 10.
    counter = Counter()
    for lamb in lambs:
        print lamb
        hists = scipy.sparse.hstack([text_hists * lamb, sift_hists * (1-lamb)]).toarray()
        preprocessing.normalize(hists, copy=False)
        s = []
        for cv in range(5):
            print cv
            xtrain, xvalid = hists[train_index], hists[valid_index]
            ytrain, yvalid = y[train_index], y[valid_index]
            for i in range(len(valid_index)):
                cos = np.dot(xtrain, xvalid[i])
                indices = cos.argsort()[::-1][:10]
                s.append(np.sum(ytrain[indices] == yvalid[i]))
        counter[lamb] = np.mean(np.array(s))
        print counter[lamb]
    print counter


def test():
    with open('data/sift_names.pkl', 'r') as f:
        names = cPickle.load(f)
    glo = glob.glob('shopping/queryimages/*.jpg')
    y = []
    for n in names:
        a = n.split('.')[0].split('_')
        y.append(a[1] + a[2])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    ytest = []
    filenames = []
    text_filenames = []
    for i, filename in enumerate(glo):
        filenames.append(filename)
        a = filename.split('.')[0].split('_')
        ytest.append(a[1] + a[2])
        filename = filename.replace('/images/', '/queryimages/')
        filename = filename.replace('img', 'descr')
        filename = filename.replace('.jpg', '.txt')
        text_filenames.append(filename)
    ytest = le.fit_transform(ytest)

    with open('data/vectorizer.pkl', 'r') as f:
        vectorizer = cPickle.load(f)
    text_histogram = vectorizer.transform(text_filenames).tocsr()
    preprocessing.normalize(text_histogram, copy=False)

    with open('data/kmean_model.pkl', 'r') as f:
        model = cPickle.load(f)
    sift = cv2.xfeatures2d.SIFT_create()
    sift_histogram = np.zeros((len(filenames), n_clusters))
    for i in range(len(filenames)):
        filename = filenames[i]
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        v = model.predict(des)
        sift_histogram[i] = np.histogram(v, bins=n_clusters, range=(0, n_clusters))[0]

    lamb = .5
    histogram = scipy.sparse.hstack([text_histogram * lamb, sift_histogram * (1-lamb)]).toarray()
    preprocessing.normalize(histogram, copy=False)

    with open('data/lshforest_combine.pkl', 'r') as f:
        lsh = cPickle.load(f)
    indices = lsh.kneighbors(histogram, n_neighbors=10)[1]
    s = []

    t = Counter()
    c = Counter()
    for i in range(len(indices)):
        s.append(np.sum(y[indices[i]] == ytest[i]))
        c[ytest[i]] += np.sum(y[indices[i]] == ytest[i])
        t[ytest[i]] += 1
    for k, v in c.most_common():
        c[k] /= float(t[k])
    print np.mean(np.array(s))  # 6.5/10
    print c.most_common()
    print le.inverse_transform(t.keys())


if __name__ == "__main__":
    if len(os.listdir('shopping/grayscale32x32')) == 0:
        tiny()
    if not os.path.exists('hists/hists.npy'):
        hist()
    if not os.path.exists('data/sift_features.pkl'):
        sift_features()
    if not os.path.exists('data/sift_leaders.pkl'):
        sift_kmean()
    if not os.path.exists('data/sift_hist.pkl'):
        sift_hist()
    if not os.path.exists('data/text_hist.pkl'):
        text_hist()
    test()