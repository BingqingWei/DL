__author__ = 'Bingqing Wei'

from keras.layers import Dense, Input, Conv2D, Conv1D, MaxPool2D, Add, ReLU
import tensorflow as tf

def residual_block_2d(prev, filters, kernel_size):
    x = ReLU()(prev)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = Add()([x, prev])
    return x

def residual_block_1d(prev, filters, kernel_size):
    x = ReLU()(prev)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
    x = Add()([x, prev])
    return x

# TODO this solution only works with 1d tensor, for tensor with shape (sequence_length, feature_size) it won't work
def to_flattened_upperTriangle(tensor):
    tensor = tf.expand_dims(tensor, axis=-1)
    dot = tf.matmul(tensor, tf.transpose(tensor))
    ones = tf.ones_like(dot)
    mask_a = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    upper_triangular_flat = tf.boolean_mask(dot, mask)
    return upper_triangular_flat
