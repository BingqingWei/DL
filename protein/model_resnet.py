__author__ = 'Bingqing Wei'
import tensorflow as tf
import numpy as np
import os
from keras.layers import Concatenate, Embedding
from keras import activations
import keras
from sklearn.model_selection import train_test_split

from lib import *
from data import *

def build_model():
    """
    :return: msa_tensor, amino_tensor, secondary_tensor, target_distance, target_torsion, train_op
    """
    msa_tensor = Input(shape=(None, 21))
    amino_tensor = Input(shape=(None,))
    secondary_tensor = Input(shape=(None,))

    amino_embed = Embedding(input_dim=21, output_dim=3)(amino_tensor)
    secondary_embed = Embedding(input_dim=8, output_dim=3)(secondary_tensor)

    x = Concatenate(axis=-1)([msa_tensor, amino_embed, secondary_embed])

    # reduce number of channels
    x = Conv1D(filters=16, kernel_size=(17,), activation=activations.relu, padding='same')(x)
    for i in range(6):
        x = residual_block_1d(x, filters=16, kernel_size=(17,))
    encoded = x
    x = pairwise_expand(encoded)

    x = Conv2D(filters=16, kernel_size=(5, 5), activation=activations.relu, padding='same')(x)
    for i in range(10):
        x = residual_block_2d(x, filters=16, kernel_size=(5, 5))
    distance_predict = Conv2D(filters=1, kernel_size=(5, 5), activation=activations.relu)(x)
    distance_predict = ToSymmetric()(distance_predict)

    x = encoded
    for i in range(4):
        x = residual_block_1d(x, filters=16, kernel_size=(17,))
    torsion_predict = Conv1D(filters=2, kernel_size=(17,), activation=activations.sigmoid)(x)

    model = keras.models.Model(inputs=[amino_tensor, secondary_tensor, msa_tensor],
                               outputs=[distance_predict, torsion_predict])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss={
                      distance_predict: keras.losses.mse,
                      torsion_predict: keras.losses.mse
                  })
    return model


def train(nb_epoch=100):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    model = build_model()
    loader = load_loader(mode='train')
    X, Y = loader.X, loader.Y
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15, random_state=2019)

    model.fit(x=train_X, y=train_Y, nb_epoch=nb_epoch,
              validation_data=[test_X, test_Y], verbose=1)


if __name__ == '__main__':
    train()




