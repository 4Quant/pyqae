import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np

__doc__ = """Custom loss functions for better dealing with unbalanced
classes and medical segmentation problems"""

def wdice_coef(weight, flip = False, smooth = 0.5):
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
    array([[ 0.52]])
    >>> keval(wdice_coef(weight = 0.1)(gt_vec, pred_vec))
    array([[ 0.11]])
    >>> keval(wdice_coef(weight = 10)(gt_vec, pred_vec))
    array([[ 1.53]])
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
        return (-1 if flip else 1)*(2. * weight * K.dot(y_true_f, K.transpose(y_pred_f)) + smooth) / ((weight*K.sum(y_true_f)) + K.sum(y_pred_f) + smooth)
    return cdice_score

import os
def _setup_keras_backend(dec = None):
    """
    Utility function for setting up Keras with tensorflow correctly
    :param dec: decimal places to keep
    :return:
    """

    os.environ['keras_backend'] = "tensorflow"
    from keras import backend as K
    if dec is None:
        return K.get_session().run
    else:
        return lambda x: (K.get_session().run(x)*np.power(10,dec)).astype(
            int)/np.power(10,dec)

def wmae_loss(pos_penalty = 1.0, neg_penalty = 1.0):
    """
    Allows for the positive (overshoot of predictions) penalty to be rescaled
    >>> keval = _setup_keras_backend(2)
    >>> gt_vec = np.zeros((5,))
    >>> pred_vec = np.linspace(-1, 1, 5)
    >>> keval(wmae_loss()(gt_vec, pred_vec))
    array([ 1.  ,  0.5 ,  0.  ,  0.33,  0.5 ])
    >>> keval(wmae_loss(pos_penalty = 0.1)(gt_vec, pred_vec))
    array([ 1.  ,  0.5 ,  0.  ,  0.03,  0.05])
    >>> keval(wmae_loss(neg_penalty = 0.1)(gt_vec, pred_vec))
    array([ 1.  ,  0.5 ,  0.  ,  0.83,  0.9 ])
    """
    def wmae(y_true, y_predict):
        diff = y_true-y_predict
        abs_diff = K.abs(diff)
        neg_diff = (diff + abs_diff)/2.0
        pos_diff = (abs_diff - diff)/2.0
        return (pos_penalty*pos_diff+neg_penalty*neg_diff)/(pos_diff+neg_penalty)
    return wmae

def show_loss(loss_history):
    epich = np.cumsum(np.concatenate([np.linspace(0.5,1,len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize = (20,5))
    _ = ax1.plot(epich,np.concatenate([mh.history['loss'] for mh in loss_history]),'b-',
               epich,np.concatenate([mh.history['val_loss'] for mh in loss_history]),'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax3.semilogy(epich,np.concatenate([mh.history['mean_squared_error'] for mh in loss_history]),'b-',
                     epich,np.concatenate([mh.history['val_mean_squared_error'] for mh in loss_history]),'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('MSE')

    _ = ax2.plot(epich,np.concatenate([mh.history['cdice_score'] for mh in loss_history]),'b-',
        epich,np.concatenate([mh.history['val_cdice_score'] for mh in loss_history]),'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Dice Coefficient')

    _ = ax4.semilogy(epich,np.concatenate([mh.history['binary_crossentropy'] for mh in loss_history]),'b-',
                     epich,np.concatenate([mh.history['val_binary_crossentropy'] for mh in loss_history]),'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('binary_crossentropy')