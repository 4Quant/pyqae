import keras.backend as K
import numpy as np
from keras.engine import Layer
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dropout, Convolution2D, MaxPooling2D
)
from keras.models import Model

__doc__ = """
A few useful layers and blocks from the fractalnet model (originally posted 
(https://github.com/snf/keras-fractalnet/blob/master/src/fractalnet.py)"""
if K._BACKEND == 'theano':
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
if K._BACKEND == 'tensorflow':
    import tensorflow as tf


def theano_multinomial(n, pvals, seed):
    rng = RandomStreams(seed)
    return rng.multinomial(n=n, pvals=pvals, dtype='float32')


def tensorflow_categorical(count, seed):
    assert count > 0
    arr = [1.] + [.0 for _ in range(count - 1)]
    return tf.random_shuffle(arr, seed)


def rand_one_in_array(count, seed=None):
    """
    Returns a random array [x0, x1, ...xn] where one is 1 and the others
    :param count:
    :param seed:
    :return:
    >>> rand_one_in_array(2, 2017)
    <tf.Tensor 'RandomShuffle:0' shape=(2,) dtype=float32>
    """
    if seed is None:
        seed = np.random.randint(1, 10e6)
    if K._BACKEND == 'theano':
        pvals = np.array([[1. / count for _ in range(count)]], dtype='float32')
        return theano_multinomial(n=1, pvals=pvals, seed=seed)[0]
    elif K._BACKEND == 'tensorflow':
        return tensorflow_categorical(count=count, seed=seed)
    else:
        raise Exception('Backend: {} not implemented'.format(K._BACKEND))


class JoinLayer(Layer):
    '''
    This layer will behave as Merge(mode='ave') during testing but
    during training it will randomly select between using local or
    global droppath and apply the average of the paths alive after
    aplying the drops.
    - Global: use the random shared tensor to select the paths.
    - Local: sample a random tensor to select the paths.
    '''

    def __init__(self, drop_p, is_global, global_path, force_path, **kwargs):
        # print "init"
        self.p = 1. - drop_p
        self.is_global = is_global
        self.global_path = global_path
        self.uses_learning_phase = True
        self.force_path = force_path
        super(JoinLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # print("build")
        self.average_shape = list(input_shape[0])[1:]

    def _random_arr(self, count, p):
        return K.random_binomial((count,), p=p)

    def _arr_with_one(self, count):
        return rand_one_in_array(count=count)

    def _gen_local_drops(self, count, p):
        # Create a local droppath with at least one path
        arr = self._random_arr(count, p)
        drops = K.switch(
            K.any(arr),
            arr,
            self._arr_with_one(count)
        )
        return drops

    def _gen_global_path(self, count):
        return self.global_path[:count]

    def _drop_path(self, inputs):
        count = len(inputs)
        drops = K.switch(
            self.is_global,
            self._gen_global_path(count),
            self._gen_local_drops(count, self.p)
        )
        ave = K.zeros(shape=self.average_shape)
        for i in range(0, count):
            ave += inputs[i] * drops[i]
        sum = K.sum(drops)
        # Check that the sum is not 0 (global droppath can make it
        # 0) to avoid divByZero
        ave = K.switch(
            K.not_equal(sum, 0.),
            ave / sum,
            ave)
        return ave

    def _ave(self, inputs):
        ave = inputs[0]
        for input in inputs[1:]:
            ave += input
        ave /= len(inputs)
        return ave

    def call(self, inputs, mask=None):
        # print("call")
        if self.force_path:
            output = self._drop_path(inputs)
        else:
            output = K.in_train_phase(self._drop_path(inputs),
                                      self._ave(inputs))
        return output

    def get_output_shape_for(self, input_shape):
        # print("get_output_shape_for", input_shape)
        return input_shape[0]


class JoinLayerGen:
    """
    JoinLayerGen will initialize seeds for both global droppath
    switch and global droppout path.
    These seeds will be used to create the random tensors that the
    children layers will use to know if they must use global droppout
    and which path to take in case it is.
    >>> from keras.layers import Input
    >>> a_in = Input(shape = (4,), name = 'a')
    >>> b_in = Input(shape = (4,), name = 'b')
    >>> join_gen = JoinLayerGen(width=4, global_p=0.5)
    >>> c_join = join_gen.get_join_layer(0.5)([a_in, b_in])
    >>> from keras.models import Model
    >>> out_model = Model(inputs = [a_in, b_in], outputs = [c_join])
    >>> out_model.summary() # doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    a (InputLayer)                   (None, 4)             0
    ____________________________________________________________________________________________________
    b (InputLayer)                   (None, 4)             0
    ____________________________________________________________________________________________________
    join_layer_1 (JoinLayer)         [(None, 4), (None, 4) 0           a[0][0]
                                                                       b[0][0]
    ====================================================================================================
    Total params: 0
    Trainable params: 0
    Non-trainable params: 0
    ____________________________________________________________________________________________________

    >>> out_model.predict([np.zeros((1,4)), np.ones((1,4))])
    array([[ 0.5,  0.5,  0.5,  0.5]], dtype=float32)
    """

    def __init__(self, width, global_p=0.5, deepest=False):
        self.global_p = global_p
        self.width = width
        self.switch_seed = np.random.randint(1, 10e6)
        self.path_seed = np.random.randint(1, 10e6)
        self.deepest = deepest
        if deepest:
            self.is_global = K.variable(1.)
            self.path_array = K.variable([1.] + [.0 for _ in range(width - 1)])
        else:
            self.is_global = self._build_global_switch()
            self.path_array = self._build_global_path_arr()

    def _build_global_path_arr(self):
        # The path the block will take when using global droppath
        return rand_one_in_array(seed=self.path_seed, count=self.width)

    def _build_global_switch(self):
        # A randomly sampled tensor that will signal if the batch
        # should use global or local droppath
        return K.equal(
            K.random_binomial((), p=self.global_p, seed=self.switch_seed), 1.)

    def get_join_layer(self, drop_p):
        global_switch = self.is_global
        global_path = self.path_array
        return JoinLayer(drop_p=drop_p, is_global=global_switch,
                         global_path=global_path, force_path=self.deepest)


def fractal_conv(filter, nb_row, nb_col, dropout=None):
    def f(prev):
        conv = prev
        conv = Convolution2D(filter, nb_row=nb_col, nb_col=nb_col,
                             init='he_normal', border_mode='same')(conv)
        if dropout:
            conv = Dropout(dropout)(conv)
        conv = BatchNormalization(mode=0, axis=1)(conv)
        conv = Activation('relu')(conv)
        return conv

    return f


# XXX_ It's not clear when to apply Dropout, the paper cited
# (arXiv:1511.07289) uses it in the last layer of each stack but in
# the code gustav published it is in each convolution block so I'm
# copying it.
def fractal_block(join_gen, c, filter, nb_col, nb_row, drop_p, dropout=None):
    def f(z):
        columns = [[z] for _ in range(c)]
        last_row = 2 ** (c - 1) - 1
        for row in range(2 ** (c - 1)):
            t_row = []
            for col in range(c):
                prop = 2 ** (col)
                # Add blocks
                if (row + 1) % prop == 0:
                    t_col = columns[col]
                    t_col.append(fractal_conv(filter=filter,
                                              nb_col=nb_col,
                                              nb_row=nb_row,
                                              dropout=dropout)(t_col[-1]))
                    t_row.append(col)
            # Merge (if needed)
            if len(t_row) > 1:
                merging = [columns[x][-1] for x in t_row]
                merged = join_gen.get_join_layer(drop_p=drop_p)(merging)
                for i in t_row:
                    columns[i].append(merged)
        return columns[0][-1]

    return f


def fractal_net(b, c, conv, drop_path, global_p=0.5, dropout=None,
                deepest=False):
    '''
    Return a function that builds the Fractal part of the network
    respecting keras functional model.
    When deepest is set, we build the entire network but set droppath
    to global and the Join masks to [1., 0... 0.] so only the deepest
    column is always taken.
    We don't add the softmax layer here nor build the model.
    >>> conv = [(64, 3, 3), (128, 3, 3), (256, 3, 3), (512, 3, 3), (512, 2, 2)]
    >>> dropout = [0., 0.1, 0.2, 0.3, 0.4]
    >>> o_net = fractal_net(c=3, b=5, conv=conv,drop_path=0.15, dropout=dropout,deepest=False)
    >>> input = Input(shape=(32, 32, 3), name = 'test_input')
    >>> output = o_net(input)
    >>> t_model = Model(inputs = [input], outputs = [output])
    >>> t_model.summary() # doctest: +NORMALIZE_WHITESPACE
    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to
    ====================================================================================================
    test_input (InputLayer)          (None, 32, 32, 3)     0
    ____________________________________________________________________________________________________
    conv2d_1 (Conv2D)                (None, 32, 32, 64)    1792        test_input[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_1 (BatchNorm (None, 32, 32, 64)    128         conv2d_1[0][0]
    ____________________________________________________________________________________________________
    activation_1 (Activation)        (None, 32, 32, 64)    0           batch_normalization_1[0][0]
    ____________________________________________________________________________________________________
    conv2d_2 (Conv2D)                (None, 32, 32, 64)    36928       activation_1[0][0]
    ____________________________________________________________________________________________________
    conv2d_3 (Conv2D)                (None, 32, 32, 64)    1792        test_input[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_2 (BatchNorm (None, 32, 32, 64)    128         conv2d_2[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_3 (BatchNorm (None, 32, 32, 64)    128         conv2d_3[0][0]
    ____________________________________________________________________________________________________
    activation_2 (Activation)        (None, 32, 32, 64)    0           batch_normalization_2[0][0]
    ____________________________________________________________________________________________________
    activation_3 (Activation)        (None, 32, 32, 64)    0           batch_normalization_3[0][0]
    ____________________________________________________________________________________________________
    join_layer_2 (JoinLayer)         [(None, 32, 32, 64),  0           activation_2[0][0]
                                                                       activation_3[0][0]
    ____________________________________________________________________________________________________
    conv2d_4 (Conv2D)                (None, 32, 32, 64)    36928       join_layer_2[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_4 (BatchNorm (None, 32, 32, 64)    128         conv2d_4[0][0]
    ____________________________________________________________________________________________________
    activation_4 (Activation)        (None, 32, 32, 64)    0           batch_normalization_4[0][0]
    ____________________________________________________________________________________________________
    conv2d_5 (Conv2D)                (None, 32, 32, 64)    36928       activation_4[0][0]
    ____________________________________________________________________________________________________
    conv2d_6 (Conv2D)                (None, 32, 32, 64)    36928       join_layer_2[0][0]
    ____________________________________________________________________________________________________
    conv2d_7 (Conv2D)                (None, 32, 32, 64)    1792        test_input[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_5 (BatchNorm (None, 32, 32, 64)    128         conv2d_5[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_6 (BatchNorm (None, 32, 32, 64)    128         conv2d_6[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_7 (BatchNorm (None, 32, 32, 64)    128         conv2d_7[0][0]
    ____________________________________________________________________________________________________
    activation_5 (Activation)        (None, 32, 32, 64)    0           batch_normalization_5[0][0]
    ____________________________________________________________________________________________________
    activation_6 (Activation)        (None, 32, 32, 64)    0           batch_normalization_6[0][0]
    ____________________________________________________________________________________________________
    activation_7 (Activation)        (None, 32, 32, 64)    0           batch_normalization_7[0][0]
    ____________________________________________________________________________________________________
    join_layer_3 (JoinLayer)         [(None, 32, 32, 64),  0           activation_5[0][0]
                                                                       activation_6[0][0]
                                                                       activation_7[0][0]
    ____________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)   (None, 16, 16, 64)    0           join_layer_3[0][0]
    ____________________________________________________________________________________________________
    conv2d_8 (Conv2D)                (None, 16, 16, 128)   73856       max_pooling2d_1[0][0]
    ____________________________________________________________________________________________________
    dropout_1 (Dropout)              (None, 16, 16, 128)   0           conv2d_8[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_8 (BatchNorm (None, 16, 16, 128)   64          dropout_1[0][0]
    ____________________________________________________________________________________________________
    activation_8 (Activation)        (None, 16, 16, 128)   0           batch_normalization_8[0][0]
    ____________________________________________________________________________________________________
    conv2d_9 (Conv2D)                (None, 16, 16, 128)   147584      activation_8[0][0]
    ____________________________________________________________________________________________________
    conv2d_10 (Conv2D)               (None, 16, 16, 128)   73856       max_pooling2d_1[0][0]
    ____________________________________________________________________________________________________
    dropout_2 (Dropout)              (None, 16, 16, 128)   0           conv2d_9[0][0]
    ____________________________________________________________________________________________________
    dropout_3 (Dropout)              (None, 16, 16, 128)   0           conv2d_10[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_9 (BatchNorm (None, 16, 16, 128)   64          dropout_2[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_10 (BatchNor (None, 16, 16, 128)   64          dropout_3[0][0]
    ____________________________________________________________________________________________________
    activation_9 (Activation)        (None, 16, 16, 128)   0           batch_normalization_9[0][0]
    ____________________________________________________________________________________________________
    activation_10 (Activation)       (None, 16, 16, 128)   0           batch_normalization_10[0][0]
    ____________________________________________________________________________________________________
    join_layer_4 (JoinLayer)         [(None, 16, 16, 128), 0           activation_9[0][0]
                                                                       activation_10[0][0]
    ____________________________________________________________________________________________________
    conv2d_11 (Conv2D)               (None, 16, 16, 128)   147584      join_layer_4[0][0]
    ____________________________________________________________________________________________________
    dropout_4 (Dropout)              (None, 16, 16, 128)   0           conv2d_11[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_11 (BatchNor (None, 16, 16, 128)   64          dropout_4[0][0]
    ____________________________________________________________________________________________________
    activation_11 (Activation)       (None, 16, 16, 128)   0           batch_normalization_11[0][0]
    ____________________________________________________________________________________________________
    conv2d_12 (Conv2D)               (None, 16, 16, 128)   147584      activation_11[0][0]
    ____________________________________________________________________________________________________
    conv2d_13 (Conv2D)               (None, 16, 16, 128)   147584      join_layer_4[0][0]
    ____________________________________________________________________________________________________
    conv2d_14 (Conv2D)               (None, 16, 16, 128)   73856       max_pooling2d_1[0][0]
    ____________________________________________________________________________________________________
    dropout_5 (Dropout)              (None, 16, 16, 128)   0           conv2d_12[0][0]
    ____________________________________________________________________________________________________
    dropout_6 (Dropout)              (None, 16, 16, 128)   0           conv2d_13[0][0]
    ____________________________________________________________________________________________________
    dropout_7 (Dropout)              (None, 16, 16, 128)   0           conv2d_14[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_12 (BatchNor (None, 16, 16, 128)   64          dropout_5[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_13 (BatchNor (None, 16, 16, 128)   64          dropout_6[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_14 (BatchNor (None, 16, 16, 128)   64          dropout_7[0][0]
    ____________________________________________________________________________________________________
    activation_12 (Activation)       (None, 16, 16, 128)   0           batch_normalization_12[0][0]
    ____________________________________________________________________________________________________
    activation_13 (Activation)       (None, 16, 16, 128)   0           batch_normalization_13[0][0]
    ____________________________________________________________________________________________________
    activation_14 (Activation)       (None, 16, 16, 128)   0           batch_normalization_14[0][0]
    ____________________________________________________________________________________________________
    join_layer_5 (JoinLayer)         [(None, 16, 16, 128), 0           activation_12[0][0]
                                                                       activation_13[0][0]
                                                                       activation_14[0][0]
    ____________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)   (None, 8, 8, 128)     0           join_layer_5[0][0]
    ____________________________________________________________________________________________________
    conv2d_15 (Conv2D)               (None, 8, 8, 256)     295168      max_pooling2d_2[0][0]
    ____________________________________________________________________________________________________
    dropout_8 (Dropout)              (None, 8, 8, 256)     0           conv2d_15[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_15 (BatchNor (None, 8, 8, 256)     32          dropout_8[0][0]
    ____________________________________________________________________________________________________
    activation_15 (Activation)       (None, 8, 8, 256)     0           batch_normalization_15[0][0]
    ____________________________________________________________________________________________________
    conv2d_16 (Conv2D)               (None, 8, 8, 256)     590080      activation_15[0][0]
    ____________________________________________________________________________________________________
    conv2d_17 (Conv2D)               (None, 8, 8, 256)     295168      max_pooling2d_2[0][0]
    ____________________________________________________________________________________________________
    dropout_9 (Dropout)              (None, 8, 8, 256)     0           conv2d_16[0][0]
    ____________________________________________________________________________________________________
    dropout_10 (Dropout)             (None, 8, 8, 256)     0           conv2d_17[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_16 (BatchNor (None, 8, 8, 256)     32          dropout_9[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_17 (BatchNor (None, 8, 8, 256)     32          dropout_10[0][0]
    ____________________________________________________________________________________________________
    activation_16 (Activation)       (None, 8, 8, 256)     0           batch_normalization_16[0][0]
    ____________________________________________________________________________________________________
    activation_17 (Activation)       (None, 8, 8, 256)     0           batch_normalization_17[0][0]
    ____________________________________________________________________________________________________
    join_layer_6 (JoinLayer)         [(None, 8, 8, 256), ( 0           activation_16[0][0]
                                                                       activation_17[0][0]
    ____________________________________________________________________________________________________
    conv2d_18 (Conv2D)               (None, 8, 8, 256)     590080      join_layer_6[0][0]
    ____________________________________________________________________________________________________
    dropout_11 (Dropout)             (None, 8, 8, 256)     0           conv2d_18[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_18 (BatchNor (None, 8, 8, 256)     32          dropout_11[0][0]
    ____________________________________________________________________________________________________
    activation_18 (Activation)       (None, 8, 8, 256)     0           batch_normalization_18[0][0]
    ____________________________________________________________________________________________________
    conv2d_19 (Conv2D)               (None, 8, 8, 256)     590080      activation_18[0][0]
    ____________________________________________________________________________________________________
    conv2d_20 (Conv2D)               (None, 8, 8, 256)     590080      join_layer_6[0][0]
    ____________________________________________________________________________________________________
    conv2d_21 (Conv2D)               (None, 8, 8, 256)     295168      max_pooling2d_2[0][0]
    ____________________________________________________________________________________________________
    dropout_12 (Dropout)             (None, 8, 8, 256)     0           conv2d_19[0][0]
    ____________________________________________________________________________________________________
    dropout_13 (Dropout)             (None, 8, 8, 256)     0           conv2d_20[0][0]
    ____________________________________________________________________________________________________
    dropout_14 (Dropout)             (None, 8, 8, 256)     0           conv2d_21[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_19 (BatchNor (None, 8, 8, 256)     32          dropout_12[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_20 (BatchNor (None, 8, 8, 256)     32          dropout_13[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_21 (BatchNor (None, 8, 8, 256)     32          dropout_14[0][0]
    ____________________________________________________________________________________________________
    activation_19 (Activation)       (None, 8, 8, 256)     0           batch_normalization_19[0][0]
    ____________________________________________________________________________________________________
    activation_20 (Activation)       (None, 8, 8, 256)     0           batch_normalization_20[0][0]
    ____________________________________________________________________________________________________
    activation_21 (Activation)       (None, 8, 8, 256)     0           batch_normalization_21[0][0]
    ____________________________________________________________________________________________________
    join_layer_7 (JoinLayer)         [(None, 8, 8, 256), ( 0           activation_19[0][0]
                                                                       activation_20[0][0]
                                                                       activation_21[0][0]
    ____________________________________________________________________________________________________
    max_pooling2d_3 (MaxPooling2D)   (None, 4, 4, 256)     0           join_layer_7[0][0]
    ____________________________________________________________________________________________________
    conv2d_22 (Conv2D)               (None, 4, 4, 512)     1180160     max_pooling2d_3[0][0]
    ____________________________________________________________________________________________________
    dropout_15 (Dropout)             (None, 4, 4, 512)     0           conv2d_22[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_22 (BatchNor (None, 4, 4, 512)     16          dropout_15[0][0]
    ____________________________________________________________________________________________________
    activation_22 (Activation)       (None, 4, 4, 512)     0           batch_normalization_22[0][0]
    ____________________________________________________________________________________________________
    conv2d_23 (Conv2D)               (None, 4, 4, 512)     2359808     activation_22[0][0]
    ____________________________________________________________________________________________________
    conv2d_24 (Conv2D)               (None, 4, 4, 512)     1180160     max_pooling2d_3[0][0]
    ____________________________________________________________________________________________________
    dropout_16 (Dropout)             (None, 4, 4, 512)     0           conv2d_23[0][0]
    ____________________________________________________________________________________________________
    dropout_17 (Dropout)             (None, 4, 4, 512)     0           conv2d_24[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_23 (BatchNor (None, 4, 4, 512)     16          dropout_16[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_24 (BatchNor (None, 4, 4, 512)     16          dropout_17[0][0]
    ____________________________________________________________________________________________________
    activation_23 (Activation)       (None, 4, 4, 512)     0           batch_normalization_23[0][0]
    ____________________________________________________________________________________________________
    activation_24 (Activation)       (None, 4, 4, 512)     0           batch_normalization_24[0][0]
    ____________________________________________________________________________________________________
    join_layer_8 (JoinLayer)         [(None, 4, 4, 512), ( 0           activation_23[0][0]
                                                                       activation_24[0][0]
    ____________________________________________________________________________________________________
    conv2d_25 (Conv2D)               (None, 4, 4, 512)     2359808     join_layer_8[0][0]
    ____________________________________________________________________________________________________
    dropout_18 (Dropout)             (None, 4, 4, 512)     0           conv2d_25[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_25 (BatchNor (None, 4, 4, 512)     16          dropout_18[0][0]
    ____________________________________________________________________________________________________
    activation_25 (Activation)       (None, 4, 4, 512)     0           batch_normalization_25[0][0]
    ____________________________________________________________________________________________________
    conv2d_26 (Conv2D)               (None, 4, 4, 512)     2359808     activation_25[0][0]
    ____________________________________________________________________________________________________
    conv2d_27 (Conv2D)               (None, 4, 4, 512)     2359808     join_layer_8[0][0]
    ____________________________________________________________________________________________________
    conv2d_28 (Conv2D)               (None, 4, 4, 512)     1180160     max_pooling2d_3[0][0]
    ____________________________________________________________________________________________________
    dropout_19 (Dropout)             (None, 4, 4, 512)     0           conv2d_26[0][0]
    ____________________________________________________________________________________________________
    dropout_20 (Dropout)             (None, 4, 4, 512)     0           conv2d_27[0][0]
    ____________________________________________________________________________________________________
    dropout_21 (Dropout)             (None, 4, 4, 512)     0           conv2d_28[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_26 (BatchNor (None, 4, 4, 512)     16          dropout_19[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_27 (BatchNor (None, 4, 4, 512)     16          dropout_20[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_28 (BatchNor (None, 4, 4, 512)     16          dropout_21[0][0]
    ____________________________________________________________________________________________________
    activation_26 (Activation)       (None, 4, 4, 512)     0           batch_normalization_26[0][0]
    ____________________________________________________________________________________________________
    activation_27 (Activation)       (None, 4, 4, 512)     0           batch_normalization_27[0][0]
    ____________________________________________________________________________________________________
    activation_28 (Activation)       (None, 4, 4, 512)     0           batch_normalization_28[0][0]
    ____________________________________________________________________________________________________
    join_layer_9 (JoinLayer)         [(None, 4, 4, 512), ( 0           activation_26[0][0]
                                                                       activation_27[0][0]
                                                                       activation_28[0][0]
    ____________________________________________________________________________________________________
    max_pooling2d_4 (MaxPooling2D)   (None, 2, 2, 512)     0           join_layer_9[0][0]
    ____________________________________________________________________________________________________
    conv2d_29 (Conv2D)               (None, 2, 2, 512)     1049088     max_pooling2d_4[0][0]
    ____________________________________________________________________________________________________
    dropout_22 (Dropout)             (None, 2, 2, 512)     0           conv2d_29[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_29 (BatchNor (None, 2, 2, 512)     8           dropout_22[0][0]
    ____________________________________________________________________________________________________
    activation_29 (Activation)       (None, 2, 2, 512)     0           batch_normalization_29[0][0]
    ____________________________________________________________________________________________________
    conv2d_30 (Conv2D)               (None, 2, 2, 512)     1049088     activation_29[0][0]
    ____________________________________________________________________________________________________
    conv2d_31 (Conv2D)               (None, 2, 2, 512)     1049088     max_pooling2d_4[0][0]
    ____________________________________________________________________________________________________
    dropout_23 (Dropout)             (None, 2, 2, 512)     0           conv2d_30[0][0]
    ____________________________________________________________________________________________________
    dropout_24 (Dropout)             (None, 2, 2, 512)     0           conv2d_31[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_30 (BatchNor (None, 2, 2, 512)     8           dropout_23[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_31 (BatchNor (None, 2, 2, 512)     8           dropout_24[0][0]
    ____________________________________________________________________________________________________
    activation_30 (Activation)       (None, 2, 2, 512)     0           batch_normalization_30[0][0]
    ____________________________________________________________________________________________________
    activation_31 (Activation)       (None, 2, 2, 512)     0           batch_normalization_31[0][0]
    ____________________________________________________________________________________________________
    join_layer_10 (JoinLayer)        [(None, 2, 2, 512), ( 0           activation_30[0][0]
                                                                       activation_31[0][0]
    ____________________________________________________________________________________________________
    conv2d_32 (Conv2D)               (None, 2, 2, 512)     1049088     join_layer_10[0][0]
    ____________________________________________________________________________________________________
    dropout_25 (Dropout)             (None, 2, 2, 512)     0           conv2d_32[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_32 (BatchNor (None, 2, 2, 512)     8           dropout_25[0][0]
    ____________________________________________________________________________________________________
    activation_32 (Activation)       (None, 2, 2, 512)     0           batch_normalization_32[0][0]
    ____________________________________________________________________________________________________
    conv2d_33 (Conv2D)               (None, 2, 2, 512)     1049088     activation_32[0][0]
    ____________________________________________________________________________________________________
    conv2d_34 (Conv2D)               (None, 2, 2, 512)     1049088     join_layer_10[0][0]
    ____________________________________________________________________________________________________
    conv2d_35 (Conv2D)               (None, 2, 2, 512)     1049088     max_pooling2d_4[0][0]
    ____________________________________________________________________________________________________
    dropout_26 (Dropout)             (None, 2, 2, 512)     0           conv2d_33[0][0]
    ____________________________________________________________________________________________________
    dropout_27 (Dropout)             (None, 2, 2, 512)     0           conv2d_34[0][0]
    ____________________________________________________________________________________________________
    dropout_28 (Dropout)             (None, 2, 2, 512)     0           conv2d_35[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_33 (BatchNor (None, 2, 2, 512)     8           dropout_26[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_34 (BatchNor (None, 2, 2, 512)     8           dropout_27[0][0]
    ____________________________________________________________________________________________________
    batch_normalization_35 (BatchNor (None, 2, 2, 512)     8           dropout_28[0][0]
    ____________________________________________________________________________________________________
    activation_33 (Activation)       (None, 2, 2, 512)     0           batch_normalization_33[0][0]
    ____________________________________________________________________________________________________
    activation_34 (Activation)       (None, 2, 2, 512)     0           batch_normalization_34[0][0]
    ____________________________________________________________________________________________________
    activation_35 (Activation)       (None, 2, 2, 512)     0           batch_normalization_35[0][0]
    ____________________________________________________________________________________________________
    join_layer_11 (JoinLayer)        [(None, 2, 2, 512), ( 0           activation_33[0][0]
                                                                       activation_34[0][0]
                                                                       activation_35[0][0]
    ____________________________________________________________________________________________________
    max_pooling2d_5 (MaxPooling2D)   (None, 1, 1, 512)     0           join_layer_11[0][0]
    ====================================================================================================
    Total params: 24,535,880
    Trainable params: 24,535,012
    Non-trainable params: 868
    ____________________________________________________________________________________________________
    '''

    def f(z):
        output = z
        # Initialize a JoinLayerGen that will be used to derive the
        # JoinLayers that share the same global droppath
        join_gen = JoinLayerGen(width=c, global_p=global_p, deepest=deepest)
        for i in range(b):
            (filter, nb_col, nb_row) = conv[i]
            dropout_i = dropout[i] if dropout else None
            output = fractal_block(join_gen=join_gen,
                                   c=c, filter=filter,
                                   nb_col=nb_col,
                                   nb_row=nb_row,
                                   drop_p=drop_path,
                                   dropout=dropout_i)(output)
            output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output)
        return output

    return f
