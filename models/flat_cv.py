from sklearn.base import BaseEstimator, TransformerMixin, clone
import pickle
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
import pandas as pd
from sklearn.compose import ColumnTransformer

class FlatCV(BaseEstimator, TransformerMixin):
    '''
      this is update from flat_classifier
      we want to do .fit(X, Y), where Y is multilabel, and be able to tune for each column of y
      this solves the issue of 
        1) using gridsearch for each estimator on the column transfomer
      this version improves upon the previous one on:
        1) directly handle pipeline with column transformer
        2) can handle pipeline with multiple steps 
      the search strategy is as follows:
        1) generate the splitting slices and cached matrix 
        2) at each step, if it's groupsearchcv, first fit on the cached X,Y to get the best_params
           then refit on X to get the best estimator
           then update the pipeline 
      shortcomings:
        1) at step 1, there're some waste of memory by vstacking, which can be improved by manually recording the cv_result
        2) there're some data leakage in the search strategy because every step has seen the entire dataset, but seems not that severe for this project
    '''
    def __init__(self, pipeline, cols, param_grids,
                 splitter = KFold(n_splits=5), 
                 other_params = {'n_jobs':1, 'refit':False, 'verbose':3, 'scoring':['f1','accuracy','f1_macro']},
                 best_scorer = 'f1',
                 search_class = GridSearchCV):
        self.cols = cols
        self.splitter = splitter
        self.search_class = search_class
        self.other_params = other_params
        self.param_grids = param_grids
        self.best_scorer = best_scorer
        self.pipeline = pipeline
    
    def _is_ColumnTransformer(self, estimator):
        return isinstance(estimator, ColumnTransformer)

    def _wrap_search_class(self, estimator, param_grid):
        return self.search_class(estimator,
                                 param_grid = param_grid,
                                 cv = self.slices,
                                 **self.other_params)
    
    def _keep_best(self, estimator):
        idx = np.argmax(estimator.cv_results_[f'mean_test_{self.best_scorer}'])
        best_params_ = estimator.cv_results_['params'][idx]
        estimator = clone(estimator.estimator)
        estimator.set_params(**best_params_)
        return estimator
    
    def fit(self, X, Y):
        nsteps = len(self.pipeline)
        
        # generate the splitting slices and cached matrix
        X_cached, Y_cached = [], []
        for train_idx, test_idx in self.splitter.split(X, Y):
            if isinstance(X, pd.DataFrame):
                X_cached += [X.iloc[train_idx], X.iloc[test_idx]]
            else:
                X_cached += [X[train_idx], X[test_idx]]
            Y_cached += [Y[train_idx], Y[test_idx]]
        self.slices = [(2*i,2*i+1) for i in range(self.splitter.get_n_splits())]
        print(f'checking slices: {self.slices}')
        
        # loop over each step
        for istep, ((name, step), param_grid) in enumerate(zip(self.pipeline.steps, self.param_grids)):
            
            print(f"doing step={istep}")
            
            # at each step, if it's groupsearchcv, first fit on the cached X,Y to get the best_params
            # then refit on X to get the best estimator
            # then update the pipeline
            if param_grid is None:
                step.fit(X, Y)
            else:
                if self._is_ColumnTransformer(step):
                    for icol, (name_, transformer, cols) in enumerate(step.transformers):
                        print(f"doing {icol} {name_}")
                        transformer = self._wrap_search_class(transformer, param_grid)
                        transformer.fit(X_cached, [y[:,icol] for y in Y_cached])
                        transformer = self._keep_best(transformer)
                        print(transformer.get_params())
                        transformer.fit(X, Y[:,icol])
                        step.set_params(**{name_:transformer})
                else:
                    step = self._wrap_search_class(step, param_grid)
                    step.fit(X_cached, Y_cached)
                    step = self._keep_best(step)
                    step.fit(X, Y)
                    self.pipeline.set_params(**{name:step})

            # prepare the X and X_cached for next step
            if istep == nsteps - 1:
                break
            # fit and transform each split
            # be careful the fit should be done only on the train_slice
            X = step.transform(X)
            X_cached_new = []
            for train_slice, test_slice in self.slices:
                print(f'checking slice: {train_slice} {test_slice}')
                step_cloned = clone(step)
                step_cloned.fit(X_cached[train_slice], Y_cached[train_slice])
                X_cached_new.append(step_cloned.transform(X_cached[train_slice]))
                X_cached_new.append(step_cloned.transform(X_cached[test_slice]))
            X_cached = X_cached_new
            
        return self
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def save(self, filename):
        with open(filename, 'wb') as fid:
            pickle.dump(self, fid)
