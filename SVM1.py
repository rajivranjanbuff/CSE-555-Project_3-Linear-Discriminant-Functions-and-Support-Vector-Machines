#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 00:19:41 2018

@author: rajivranjan
"""
from sklearn.datasets import fetch_mldata
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC
import numpy as np


def main():
    mnist = fetch_mldata("MNIST original")
    X_all, y_all = mnist.data/255., mnist.target
    print("scaling")
    X = X_all[:60000, :]
    y = y_all[:60000]

    X_test = X_all[60000:, :]
    y_test = y_all[60000:]

    svm = SVC(cache_size=1000, kernel='linear')

    parameters = {'C':10. ** np.arange(1,5), 'gamma':2. ** np.arange(-5, -1)}
    print("grid search")
    grid = GridSearchCV(svm, parameters, cv=StratifiedKFold(y, 5), verbose=3, n_jobs=-1)
    grid.fit(X, y)
    print("predicting")
    print ("score: ", grid.score(X_test, y_test))
    print (grid.best_estimator_)

if __name__ == "__main__":
    main()