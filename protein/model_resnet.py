__author__ = 'Bingqing Wei'

import tensorflow as tf
import numpy as np
import os
from keras.layers import Concatenate

from lib import *
from data import *


def bulid_model():
    msa_tensor = tf.placeholder(shape=(1, None, 21))
    amino_tensor = tf.placeholder(shape=(1, None))
    secondary_tensor = tf.placeholder(shape=(1, None))

    x = Concatenate(axis=1)([msa_tensor, amino_tensor, secondary_tensor])
    for i in range(6):
        x = residual_block_1d(x, filters=16, kernel_size=(17,))

    # x shape: (1, sequence_length, 16)



