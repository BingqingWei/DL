__author__ = 'Bingqing Wei'
import tensorflow as tf
from data import *
from keras.models import Model
from keras.layers import *

def build_attention_block(in_l, in_r):
    x = tf.tensordot(in_l, in_r)
    x = tf.nn.softmax(x)
    x = tf.tensordot(in_l, x)
    return tf.concat([in_r, x], axis=-1)

"""
Input shape: (1, n, x) (x: ngram size, n: sequence length)
"""
def build_model(aa_input_dim, ss_input_dim):
    in1 = Embedding(aa_input_dim, 32)
    in2 = Embedding(ss_input_dim, 32)
    in3 = tf.placeholder(shape=(1, None))
    x = Concatenate()([in1, in2, in3])
    l1 = Bidirectional(LSTM(75, activation=activations.tanh))(x)
    l2 = LSTM(150, activation=activations.tanh)(l1)
    l3 = LSTM(150, activation=activations.tanh)(l2)
    l4 = LSTM(150, activation=activations.tanh)(l3)
    l5 = LSTM(150, activation=activations.tanh)(l4)

    lstms = [l1, l2, l3, l4, l5]
    attention_blocks = []
    for i in range(len(lstms)):
        for j in range(i + 1, len(lstms)):
            attention_blocks.append(build_attention_block(lstms[i], lstms[j]))
    x = Add()(attention_blocks)
    x1 = TimeDistributed(Dense(1, activation=activations.linear))(x)
    x2 = TimeDistributed()

    model = Model(inputs=[in1, in2, in3], outputs=[x])
