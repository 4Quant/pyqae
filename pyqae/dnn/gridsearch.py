import multiprocessing
import os

import numpy as np
import pandas as pd
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV, KFold
from pyqae.utils import Union

np.random.seed(2017)


def setup_environment():
    os.environ['KERAS_BACKEND'] = 'theano'
    os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True"
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())


def build_kr_model(in_model_fcn):
    model = KerasRegressor(build_fn=in_model_fcn,
                           nb_epoch=5,
                           batch_size=20,
                           verbose=2,
                           shuffle=True)
    lrs = [0.5, 1e-2]
    decays = [1e-6]
    momentums = [0.8]
    nesterovs = [True]
    layers = np.array([1, 2])
    depth = np.array([2, 8])
    dropout = np.array([0, 0.5])
    epochs = np.array([10])
    batch_size = np.array([25])
    use_bn = np.array([False, True])
    param_grid = dict(lr=lrs,
                      decay=decays,
                      momentum=momentums,
                      nesterov=nesterovs,
                      nb_epoch=epochs,
                      batch_size=batch_size,
                      dropout_rate=dropout,
                      depth=depth,
                      layers=layers,
                      use_bn=use_bn)
    grid = GridSearchCV(estimator=model,
                        cv=KFold(2),
                        param_grid=param_grid,
                        verbose=False,
                        n_jobs=1)
    return model, grid


def fit_grid(grid):
    grid_result = grid.fit()
    return grid_result, pd.DataFrame(grid_result.cv_results_)


def show_results(grid_result):
    # todo append columns
    [crow.to_dict() for _, crow in pd.DataFrame(grid_result.cv_results_).iterrows()]
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return pd.DataFrame([dict([('score_mean', scores.mean()),
                        ('score_std', scores.std())] +
                       list(params.items())) for params, mean_score, scores in grid_result.grid_scores_]).sort_values(
        'score_mean')
