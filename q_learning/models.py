from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from keras.layers import Convolution2D, Flatten, Input, Dense
from keras.models import Model

def atari_cnn(input_shape, n_actions):
    """
    input_shape: 3D Tensor (height, width, channels), TF format

    Where history_len is the number of stacked images (think video clip)

    n_actions: int

    The number of possible actions in our environment.
    """

    input = Input(shape=input_shape)
    x = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', dim_ordering='tf')(input)
    x = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu', dim_ordering='tf')(x)
    x = Convolution2D(64, 3, 3, subsample=(1,1), activation='relu', dim_ordering='tf')(x)
    x = Flatten()(x)

    hidden = Dense(512, activation='relu')(x)
    output = Dense(n_actions, activation='relu')(hidden)

    return Model(input, output)
