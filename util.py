from collections import Counter
import pandas as pd
from wrappers import *
import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import cPickle
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV


PATTERN = ['\d[\d\.,]+[\w%]+',
           '\w+']
PATTERN = '|'.join(PATTERN)
ALL_TOPICS = Counter({'xe': 14, 'thoi_su': 13, 'the_thao': 12, 'the_gioi': 11, 'tam_su': 10, 'suc_khoe': 9, 'so_hoa': 8,
                      'phap_luat': 7, 'kinh_doanh': 6, 'khoa_hoc': 5, 'giao_duc': 4, 'giai_tri': 3, 'gia_dinh': 2,
                      'du_lich': 1, 'cong_dong': 0})
INV_ALL_TOPICS = {v: k for k, v in ALL_TOPICS.most_common()}


@elapsed()
def create_all():
    """ Create CSV file contains ID and topic ID of all articles """
    if os.path.exists('data/all.csv'):
        return

    glo = glob.glob('input/*/*')

    ids = []
    topics = []
    for idx, filename in enumerate(glo):
        details = filename.split('/')
        ids.append(details[-1].split('.')[0])
        topic = details[-2]
        topics.append(ALL_TOPICS[topic])

    df = pd.DataFrame()
    df['id'] = ids
    df['topic'] = topics
    df.to_csv('data/all.csv', index=False)


@elapsed()
def create_train_test():
    """ Create train file and test file """
    if os.path.exists('data/train.csv'):
        return
    np.random.seed(20958)
    n_test = 1000
    xtrain = pd.read_csv('data/all.csv')
    # xao tron du lieu tap luyen
    xtrain = xtrain.reindex(np.random.permutation(xtrain.index))
    xtrain.reset_index(inplace=True, drop=True)

    xtest = xtrain[:n_test]
    xtrain = xtrain[n_test:]
    xtrain.to_csv('data/train.csv', index=False)
    xtest.to_csv('data/test.csv', index=False)


def preprocess(xfilename, vectorizer, return_vectorizer=False):
    if not os.path.exists(xfilename) or return_vectorizer:
        xtrain = pd.read_csv('data/train.csv')
        xtest = pd.read_csv('data/test.csv')
        xall = pd.concat([xtrain, xtest], ignore_index=True)
        filenames = []
        for i in range(len(xall)):
            filename = 'input/%s/%04d.txt' % (INV_ALL_TOPICS[xall.iloc[i].topic], xall.iloc[i].id)
            filenames += [filename]
        xall_transformed = vectorizer.fit_transform(filenames).tocsr()
        xtrain_raw = pd.read_csv('data/train.csv')
        xtrain = xall_transformed[:len(xtrain_raw)]
        xtest = xall_transformed[len(xtrain_raw):]
        with open(xfilename, 'w') as f:
            cPickle.dump([xtrain, xtest], f)
    else:
        with open(xfilename, 'r') as f:
            xtrain, xtest = cPickle.load(f)

    if return_vectorizer:
        return xtrain, xtest, vectorizer
    return xtrain, xtest


@elapsed()
def tfidf(return_vectorizer=False, ngram=2):
    vectorizer = TfidfVectorizer(input='filename', token_pattern="(?u)"+PATTERN, ngram_range=(1, ngram), norm='l2',
                                 min_df=3)
    return preprocess('data/x_tfidf_%igram.mat' % ngram, vectorizer, return_vectorizer=return_vectorizer)


@elapsed()
def count_binary(binary=False, ngram=2):
    vectorizer = CountVectorizer(input='filename', token_pattern="(?u)"+PATTERN, ngram_range=(1, ngram), binary=binary,
                                 min_df=3)
    return preprocess('data/x_cbinary_%i_%igram.mat' % (binary, ngram), vectorizer)


def tuning(xtrain, ytrain, pipeline, parameters):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    grid_search.fit(xtrain, ytrain)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def random_tuning(xtrain, ytrain, pipeline, parameters):
    grid_search = RandomizedSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, n_iter=100)
    grid_search.fit(xtrain, ytrain)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))