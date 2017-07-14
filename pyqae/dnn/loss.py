from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

from pyqae.utils import pprint

__doc__ = """Custom loss functions for better dealing with unbalanced
classes and medical segmentation problems"""


def wdice_coef(weight, flip=False, smooth=0.5):
    # type: (float, bool, float) -> Callable[(tf.Tensor, tf.Tensor), tf.Tensor]
    """
    The weighted dice function allows for very negative or positive sets so the penalty for false positives can be tweaked
    a high weight makes false positives less of an issue
    :param weight: the weight to give to the positive class (TP vs FP)
        weight = 0.1 causes FP to be strongly penalized
        weight = 10 causes FP to be lightly penalized (see doc-tests)
    :param flip: if the value should be flipped (to use as loss instead of
    score)
    :param smooth: (smoothing coefficient to avoid issues in the 0/0 case
    :return:
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = np.expand_dims(np.eye(3),0)
    >>> pred_vec = np.ones((1, 3, 3))
    >>> keval(wdice_coef(1.0)(gt_vec, pred_vec))
    [[ 0.52]]
    >>> keval(wdice_coef(weight = 0.1)(gt_vec, pred_vec))
    [[ 0.11]]
    >>> keval(wdice_coef(weight = 10)(gt_vec, pred_vec))
    [[ 1.53]]
    """

    def cdice_score(y_true, y_pred):
        # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
        """
        The custom DICE score based on the weight, flip and smooth provided
        above
        :param y_true:
        :param y_pred:
        :return:
        """
        y_true_f = K.batch_flatten(y_true)
        y_pred_f = K.batch_flatten(y_pred)
        return (-1 if flip else 1) * (
        2. * weight * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / (
               (weight * K.sum(y_true_f)) + K.sum(y_pred_f) + smooth)

    return cdice_score


def cdice_coef_loss_2d(weights, fp_weight=1.0, smooth=0.1):
    """
    Create a dice loss coefficient for multiple class segmentation (more
    than one channel)
    :param weights:
    :param fp_weight:
    :param smooth:
    :return:
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = np.expand_dims(np.stack([np.eye(3), np.zeros((3,3))],-1), 0)
    >>> pred_vec = np.ones_like(gt_vec)
    >>> loss_fun = cdice_coef_loss_2d([1, 1])
    >>> keval(loss_fun(gt_vec, pred_vec))
    -0.5151212423939696
    >>> loss_fun = cdice_coef_loss_2d([1, 0.5])
    >>> keval(loss_fun(gt_vec, pred_vec))
    -0.5096267368994641
    >>> loss_fun = cdice_coef_loss_2d([0.5, 1])
    >>> keval(loss_fun(gt_vec, pred_vec))
    -0.26305512669149034
    """

    def _temp_func(y_true, y_pred):
        assert y_pred.shape[3] == len(weights), \
            "Weight dimension does not match! {} != {}".format(y_pred.shape[3],
                                                               len(weights))
        loss_out = []
        for i, w in enumerate(weights):
            y_true_f = K.flatten(y_true[:, :, :, i])
            y_pred_f = K.flatten(y_pred[:, :, :, i])
            loss_out += [float(w) * (
                2. * float(fp_weight) * K.sum(
                    y_true_f * y_pred_f) + smooth) / (
                             float(fp_weight) * K.sum(y_true_f) + K.sum(
                                 y_pred_f) + smooth)]

        return -1 * reduce(lambda a, b: a + b, loss_out)

    return _temp_func

def simple_dice(y_true, y_pred):
    """
    A simple DICE implementation without any weighting
    :param y_true:
    :param y_pred:
    :return:
    >>> import tensorflow as tf
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = np.expand_dims(np.stack([np.eye(3), np.zeros((3,3))],-1), 0)
    >>> pred_vec = tf.constant(np.ones_like(gt_vec))
    >>> keval(simple_dice(gt_vec, pred_vec))
    0.28571428435374147
    """
    y_t = K.batch_flatten(y_true)
    y_p = K.batch_flatten(y_pred)
    return 2.0*K.sum(y_t*y_p) / (K.sum(y_t)+K.sum(y_p)+K.epsilon())

def simple_dice_loss(y_true, y_pred):
    """
    A simple inverted dice to use as a loss function
    :param y_true:
    :param y_pred:
    :return:
    """
    return 1-simple_dice(y_true, y_pred)

def true_positives(y_true, y_pred):
    y_t = K.batch_flatten(y_true)
    y_p = K.batch_flatten(y_pred)
    return K.sum(y_t*y_p)

def true_negatives(y_true, y_pred):
    y_t = K.batch_flatten(1-y_true)
    y_p = K.batch_flatten(1-y_pred)
    return K.sum(y_t*y_p)

def ppv(y_true, y_pred):
    """
    Positive predictive value
    :param y_true:
    :param y_pred:
    :return:
    >>> import tensorflow as tf
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = tf.constant(np.eye(5))
    >>> pred_vec = tf.constant(np.ones((5,5)))
    >>> keval(ppv(gt_vec, pred_vec))
    0.24999999874999998
    >>> pred_vec = tf.constant(np.zeros((5,5)))
    >>> keval(ppv(gt_vec, pred_vec))
    0.0
    """
    return true_positives(y_true, y_pred) / (K.sum(y_true) + K.epsilon())

def npv(y_true, y_pred):
    """
    Negative predictive value
    :param y_true:
    :param y_pred:
    :return:
    >>> import tensorflow as tf
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = tf.constant(np.eye(5))
    >>> pred_vec = tf.constant(np.ones((5,5)))
    >>> keval(ppv(gt_vec, pred_vec))
    0.24999999874999998
    >>> pred_vec = tf.constant(np.zeros((5,5)))
    >>> keval(ppv(gt_vec, pred_vec))
    0.0
    """
    return true_negatives(y_true, y_pred) / (K.sum(1-y_true)+K.epsilon())

def _setup_keras_backend(dec=None):
    """
    Utility function for setting up Keras with tensorflow correctly
    :param dec: decimal places to keep
    :return:
    """
    import os
    os.environ['keras_backend'] = "tensorflow"
    from keras import backend as K
    return lambda x: pprint(K.get_session().run(x),
                            p=2 if dec is None else dec)


def wmae_loss(pos_penalty=1.0, neg_penalty=1.0):
    """
    Allows for the positive (overshoot of predictions) penalty to be rescaled
    >>> import tensorflow as tf
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = tf.constant(np.zeros((5,)))
    >>> pred_vec = tf.constant(np.linspace(-1, 1, 5))
    >>> keval(wmae_loss()(gt_vec, pred_vec))
    [ 1.    0.5   0.    0.33  0.5 ]
    >>> keval(wmae_loss(pos_penalty = 0.1)(gt_vec, pred_vec))
    [ 1.    0.5   0.    0.03  0.05]
    >>> keval(wmae_loss(neg_penalty = 0.1)(gt_vec, pred_vec))
    [ 1.    0.5   0.    0.83  0.91]
    """

    def wmae(y_true, y_predict):
        diff = y_true - y_predict
        abs_diff = K.abs(diff)
        neg_diff = (diff + abs_diff) / 2.0
        pos_diff = (abs_diff - diff) / 2.0
        return (pos_penalty * pos_diff + neg_penalty * neg_diff) / (
        pos_diff + neg_penalty)

    return wmae


def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax3.semilogy(epich, np.concatenate(
        [mh.history['mean_squared_error'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_mean_squared_error'] for mh in loss_history]),
                     'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('MSE')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['cdice_score'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_cdice_score'] for mh in loss_history]), 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Dice Coefficient')

    _ = ax4.semilogy(epich, np.concatenate(
        [mh.history['binary_crossentropy'] for mh in loss_history]), 'b-',
                     epich, np.concatenate(
            [mh.history['val_binary_crossentropy'] for mh in loss_history]),
                     'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('binary_crossentropy')


from keras.callbacks import Callback


class CyclicLR(Callback):
    """
    This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitute of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    >>> clr_triangular = CyclicLR(mode='triangular')
    >>> clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular2')
    >>> from keras.models import Sequential
    >>> from keras.layers import Dense
    >>> t_model = Sequential()
    >>> t_model.add(Dense(4, input_shape = (12,), name = 'Dense'))
    >>> t_model.compile(optimizer = 'sgd', loss = 'mse', callbacks = [clr])
    >>> X, y = np.random.uniform(size = (100, 12)), np.random.uniform(size = (100, 4))
    >>> t_model.fit(X,y,nb_epoch=2)
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000.,
                 mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0,
                                                                            (
                                                                                1 - x)) * self.scale_fn(
                cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0,
                                                                            (
                                                                                1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
