__author__ = 'Bingqing Wei'
import tensorflow as tf
import numpy as np
import os
from keras.layers import Concatenate, Embedding
from keras import activations
import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lib import *
from data import *

class ProteinModel:
    def __init__(self, work_dir, save_per_epoch=5,
                 per_process_gpu_memory_fraction=0.7):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.sess = tf.Session(config=config)
        self.msa_tensor, self.amino_tensor, self.secondary_tensor, \
            self.target_distance, self.target_torsion, self.train_op, self.loss = self.build_model()

        self.save_per_epoch = save_per_epoch
        self.graph_saver = tf.train.Saver()

        self.work_dir = work_dir
        self.summary_writer = tf.summary.FileWriter(logdir=work_dir, graph=self.sess.graph)

    def get_tensors(self):
        return self.msa_tensor, self.amino_tensor, self.secondary_tensor, \
            self.target_distance, self.target_torsion, self.train_op


    def build_model(self):
        """
        :return: msa_tensor, amino_tensor, secondary_tensor, target_distance, target_torsion, train_op
        """
        msa_tensor = tf.placeholder(shape=(1, None, 21), dtype=tf.float32)
        amino_tensor = tf.placeholder(shape=(1, None), dtype=tf.int32)
        secondary_tensor = tf.placeholder(shape=(1, None), dtype=tf.int32)

        amino_embed = Embedding(input_dim=21, output_dim=3)(amino_tensor)
        secondary_embed = Embedding(input_dim=8, output_dim=3)(secondary_tensor)

        x = Concatenate(axis=-1)([msa_tensor, amino_embed, secondary_embed])

        # reduce number of channels
        x = Conv1D(filters=16, kernel_size=(17,), activation=activations.relu, padding='same')(x)
        for i in range(1):
            x = residual_block_1d(x, filters=16, kernel_size=(17,))
        encoded = x
        x = pairwise_expand(encoded)

        x = Conv2D(filters=16, kernel_size=(5, 5), activation=activations.relu, padding='same')(x)
        for i in range(1):
            x = residual_block_2d(x, filters=16, kernel_size=(5, 5))
        distance_predict = Conv2D(filters=1, kernel_size=(5, 5), activation=activations.relu, padding='same')(x)
        distance_predict = to_symmetric(distance_predict)
        distance_predict = tf.squeeze(distance_predict, axis=-1)

        x = encoded
        for i in range(4):
            x = residual_block_1d(x, filters=16, kernel_size=(17,))
        torsion_predict = Conv1D(filters=2, kernel_size=(17,), activation=activations.sigmoid, padding='same')(x)

        target_distance = tf.placeholder(shape=(1, None, None), dtype=tf.float32)
        target_torsion = tf.placeholder(shape=(1, None, 2), dtype=tf.float32)

        loss = tf.losses.mean_squared_error(labels=target_distance, predictions=distance_predict) + \
            tf.losses.mean_squared_error(labels=target_torsion, predictions=torsion_predict)

        train_op = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
        return msa_tensor, amino_tensor, secondary_tensor, target_distance, target_torsion, train_op, loss

    def gen_feed_dict(self, x, y):
        feed_dict = {
            self.msa_tensor: np.expand_dims(x[2], axis=0),
            self.secondary_tensor: np.expand_dims(x[1], axis=0),
            self.amino_tensor: np.expand_dims(x[0], axis=0),

            self.target_distance: np.expand_dims(y[0], axis=0),
            self.target_torsion: np.expand_dims(y[1], axis=0)
        }
        return feed_dict


    def train(self, loader, nb_epochs=100, verbose=True):
        msa_tensor, amino_tensor, secondary_tensor, \
            target_distance, target_torsion, train_op = self.get_tensors()

        X, Y = loader.X, loader.Y
        X = X[:300]
        Y = Y[:300]
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15, random_state=2019)
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(nb_epochs):
            print('Starting Epoch-{}'.format(epoch))
            avg_loss = 0.0
            count = 0.0
            with tqdm(total=len(train_X), unit='it', unit_scale=True, unit_divisor=1024) as pbar:
                for x, y in zip(train_X, train_Y):
                    loss, _ = self.sess.run([self.loss, train_op], feed_dict=self.gen_feed_dict(x, y))
                    avg_loss += loss
                    count += 1
                    pbar.set_postfix(avg_loss=avg_loss / count)
                    pbar.update()
            self.evaluate(test_X, test_Y, global_step=epoch, verbose=verbose)
            if epoch % self.save_per_epoch == 0:
                self.graph_saver.save(sess=self.sess,
                                      save_path=os.path.join(self.work_dir, 'model.ckpt'), global_step=epoch)

    def evaluate(self, test_X, test_Y, global_step, verbose):
        avg_loss = 0.0
        count = 0.0
        for x, y in zip(test_X, test_Y):
            avg_loss += self.sess.run(self.loss, feed_dict=self.gen_feed_dict(x, y))
            count += 1
        avg_loss /= count
        summary = tf.Summary()
        summary.value.add(tag='eval_loss', simple_value=avg_loss)
        self.summary_writer.add_summary(summary=summary, global_step=global_step)
        if verbose:
            print('Epoch-{} Eval-Loss-{}'.format(global_step, avg_loss))


if __name__ == '__main__':
    loader = load_loader(mode='train')
    model = ProteinModel(work_dir='./data', per_process_gpu_memory_fraction=0.6)
    model.train(loader=loader)




