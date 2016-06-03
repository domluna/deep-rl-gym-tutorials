from __future__ import absolute_import

from keras.layers import Convolution2D, Flatten, Input, Dense, Lambda
from keras.models import Model
from keras import backend as K

def atari_cnn(input_shape, n_actions):
    """
    Follows the network architecture described in the 2015 Deepmind Nature paper.

    input_shape: 3D Tensor (channels, height, width) format
    n_actions: int
    """

    input = Input(shape=input_shape)
    x = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(input)
    x = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(x)
    x = Convolution2D(64, 3, 3, subsample=(1,1), activation='relu')(x)
    x = Flatten()(x)

    hidden = Dense(512, activation='relu')(x)
    output = Dense(n_actions)(hidden)

    return Model(input, output)

def duel_atari_cnn(input_shape, n_actions, mode='mean'):
    """
    Follows the network architecture described in the 2015 Deepmind Nature paper
    with the changes proposed in Dueling Network paper.

    input_shape: 3D Tensor (channels, height, width) format
    n_actions: int
    """

    agg = None
    if mode == 'mean':
        agg = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(n_actions,))
    elif mode == 'max':
	agg = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(n_actions,))
    else:
        raise ValueError("mode must be either 'mean' or 'max'")

    input = Input(shape=input_shape)
    x = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(input)
    x = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(x)
    x = Convolution2D(64, 3, 3, subsample=(1,1), activation='relu')(x)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dense(n_actions+1)(x)
    output = agg(x)

    return Model(input, output)





