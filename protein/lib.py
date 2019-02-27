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

def to_flattened_upperTriangle(tensor):
    """
    this solution only works with 1d tensor, for tensor with shape (sequence_length, feature_size) it won't work
    :param tensor:
    :return:
    """

    tensor = tf.expand_dims(tensor, axis=-1)
    dot = tf.matmul(tensor, tf.transpose(tensor))
    ones = tf.ones_like(dot)
    mask_a = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask
    upper_triangular_flat = tf.boolean_mask(dot, mask)
    return upper_triangular_flat

def pairwise_expand(vec):
    """
    :param vec: of shape (batch_size, N, filter_size)
    :return: of shape (batch_size, N. N. 2 * filter_size)
    """
    vec = tf.transpose(vec, perm=[0, 2, 1])
    vec = tf.expand_dims(vec, axis=3)
    ones = tf.ones(shape=(1, tf.shape(vec)[-2]))
    x = tf.einsum('abcd,de->abce', vec, ones)
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    y = tf.transpose(x, perm=[0, 2, 1, 3])
    return tf.concat([x, y], axis=-1)

def to_symmetric(matrix):
    """
    :param matrix: of shape (batch_size, N, N, d)
    """
    matrix_T = tf.transpose(matrix, perm=[0, 2, 1, 3])
    return (matrix + matrix_T) / 2.0


from keras.layers import Layer
class ToSymmetric(Layer):
    def __init__(self, **kwargs):
        super(ToSymmetric, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ToSymmetric, self).build(input_shape)

    def call(self, x):
        x = to_symmetric(x)
        return tf.squeeze(x, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]

