from sklearn.base import BaseEstimator, TransformerMixin, clone
from copy import deepcopy
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV

class FlatClassifier(BaseEstimator, TransformerMixin):
    '''
      PLEASE DO NOT CLONE ME !!
      we want to do .fit(X, Y), where Y is multilabel, and be able to tune for each column of y
      this solves the issue of 
        1) using gridsearch for each estimator
        2) can be used as a column transformer using fitted estimator
      it starts with a preprocess step, which is common for all 
    '''
    def __init__(self, baseclf, cols, preprocess=None):
        self.preprocess = preprocess
        self.baseclf = baseclf
        self.cols = cols

    def fit(self, X, Y, selector=None, selector_params=None):
        '''
           selector here is like GridsearchCV and its associated params
        '''
        if self.preprocess is not None:
            X = self.preprocess.transform(X)
        self.clfs = []
        _, n = Y.shape
        for i in range(n):
            print(f"doing {self.cols[i]}")
            y = Y[:,i]
            if selector is None:
                clf = clone(self.baseclf)
            else:
                clf = selector(self.baseclf, **selector_params)
            clf.fit(X, y)
            self.clfs.append(deepcopy(clf))

    def _keep_best(self):
        tmp = []
        for clf in self.clfs:
            if isinstance(clf, GridSearchCV):
                tmp.append(deepcopy(clf.best_estimator_))
            else:
                tmp.append(deepcopy(clf))
        self.clfs = tmp

    def refit(self, X, Y, best_only=True):
        if self.preprocess is not None:
            X = self.preprocess.transform(X)
        if best_only:
            self._keep_best()
        for i, clf in enumerate(self.clfs):
            print(f"doing {self.cols[i]}")
            y = Y[:,i]
            clf.fit(X, y)

    def predict(self, X):
        if self.preprocess is not None:
            X = self.preprocess.transform(X)
        Y = []
        for clf in self.clfs:
            Y.append(clf.predict(X).reshape(-1,1))
        return np.concatenate(Y, axis=1)

    def predict_proba(self, X):
        if self.preprocess is not None:
            X = self.preprocess.transform(X)
        Y = []
        for clf in self.clfs:
            Y.append(clf.predict_proba(X)[:,1].reshape(-1,1))
        return np.concatenate(Y, axis=1)
       
    def transform(self, X):
        return self.predict_proba(X)

    def save(self, filename, best_only=True):
        if best_only:
            self._keep_best()
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)
