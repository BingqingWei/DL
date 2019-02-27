__author__ = 'Bingqing Wei'

import tensorflow as tf
import numpy as np

def test1(sess):
    x = tf.Variable(tf.random_uniform((4,), minval=-.1, maxval=.1, dtype=tf.float64))
    sess.run(tf.global_variables_initializer())
    print(sess.run(x))
    print('#' * 20)
    print(sess.run(tf.cross(x, x)))

def test2(sess):
    prev_1d = tf.Variable(tf.random_uniform((3, 2)))
    sess.run(tf.global_variables_initializer())
    out_1d = tf.expand_dims(prev_1d, axis=2)
    ones = tf.ones((1, 3))
    # left_1d = tf.tensordot(out_1d, ones, [[3], [0]])
    left_1d = tf.einsum('abc,ce->abe', out_1d, ones)
    left_1d = tf.transpose(left_1d, perm=[0, 2, 1])
    right_1d = tf.transpose(left_1d, perm=[1, 0, 2])
    print(sess.run(left_1d))
    print(sess.run(right_1d))


def test3(sess):
    A = tf.Variable(tf.random_uniform((3, 3)))
    sess.run(tf.global_variables_initializer())
    A = tf.expand_dims(A, axis=-1)
    dot = tf.matmul(A, tf.transpose(A))
    print(sess.run(dot))

    ones = tf.ones_like(dot)
    mask_a = tf.matrix_band_part(ones, 0, -1)  # Upper triangular matrix of 0s and 1s
    mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a - mask_b, dtype=tf.bool)  # Make a bool mask

    upper_triangular_flat = tf.boolean_mask(dot, mask)
    print(sess.run(upper_triangular_flat))

def test4(sess):
    A = tf.Variable(tf.random_uniform((1, 3, 1)))
    A = tf.transpose(A, perm=[0, 2, 1])
    A = tf.expand_dims(A, axis=3)
    ones = tf.ones(shape=(1, tf.shape(A)[-2]))
    x = tf.einsum('abcd,de->abce', A, ones)
    x = tf.transpose(x, perm=[0, 2, 3, 1])
    y = tf.transpose(x, perm=[0, 2, 1, 3])
    z = x + y
    sess.run(tf.global_variables_initializer())
    print(sess.run(z))


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session()
    test4(sess)