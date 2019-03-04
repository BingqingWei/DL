__author__ = 'Bingqing Wei'
from model3_batch.data import *
from model3_batch.geom_ops import *
from model3_batch.utils import *

import sys
import numpy as np
from tensorflow.keras.layers import Concatenate, Embedding, LeakyReLU, LSTM, TimeDistributed, \
    Bidirectional, GRU, Dense
from keras import activations
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle
import keras.backend as K

class ProteinModel:
    def __init__(self, work_dir, n_words_aa, n_words_q8,
                 per_process_gpu_memory_fraction=0.7):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.sess = tf.Session(config=config)
        K.set_session(self.sess)
        self.model, self.opt = self.build_model(n_words_aa, n_words_q8)
        self.work_dir = work_dir
        self.summary_writer = tf.summary.FileWriter(logdir=work_dir, graph=self.sess.graph)
        self.saver = ModelSaver(work_dir=work_dir, model=self.model)

    def build_model(self, n_words_aa, n_words_q8):
        """
        :return msa_tensor, amino_tensor, secondary_tensor, \
               target_distance, target_torsion, train_op, train_loss, eval_loss
        """
        inputs = [Input(shape = (None, )), Input(shape = (None, )), Input(shape = (None, 21))]
        amino_embed = Embedding(input_dim=n_words_aa, output_dim=128)(inputs[0])
        secondary_embed = Embedding(input_dim=n_words_q8, output_dim=64)(inputs[1])

        x = Concatenate(axis=-1)([amino_embed, secondary_embed, inputs[2]])
        x = Bidirectional(GRU(units=64, return_sequences=True,
                              recurrent_dropout=0.1, recurrent_activation='relu'))(x)
        y = TimeDistributed(Dense(units=3, activation=activations.tanh))(x)
        model = keras.Model(inputs, y)
        model.summary()

        opt = tf.train.RMSPropOptimizer(learning_rate=0.0005)
        return model, opt

    @staticmethod
    def gen(X, Y, steps, batch_size):
        idx_shuffle = np.random.permutation(X[0].shape[0])
        for i in range(steps):
            if (i + 1) * batch_size >= len(idx_shuffle): break
            idx_batch = idx_shuffle[i * batch_size : (i+1) * batch_size]
            x_train = [data[idx_batch] for data in X]
            y_train = [data[idx_batch] for data in Y]
            yield x_train, y_train

    def compute_test_loss(self, x, y):
        batch_aa, batch_q8, batch_msa, batch_seqlen, \
            batch_dcalphas, batch_phis, batch_psis = self.decompress(x, y)
        angle_scale = 180 / np.pi  # convert angles from [-pi, pi] to [-180, 180]
        with tf.GradientTape() as tape:
            y_out = self.model([batch_aa, batch_q8, batch_msa])
            torsion_angles = np.pi * y_out
            phi, psi = torsion_angles[:, :, 0], torsion_angles[:, :, 1]
            phi_scaled, psi_scaled = phi * angle_scale, psi * angle_scale  # from [-pi, pi] to [-180, 180]

            dist_matrix, coordinates = DistanceMatrix()(torsion_angles)
            loss_all = 0.0
            loss_all += mse_torsion_angle(y_true=batch_psis, y_pred=psi_scaled, batch_seqlen=batch_seqlen)
            loss_all += mse_torsion_angle(y_true=batch_phis, y_pred=phi_scaled, batch_seqlen=batch_seqlen)
            loss_all += mse_dist_matrix(y_true=batch_dcalphas, y_pred=dist_matrix, batch_seqlen=batch_seqlen)
            return loss_all, tape

    def decompress(self, x, y):
        batch_aa, batch_q8, batch_msa, batch_seqlen = x
        batch_dcalphas, batch_phis, batch_psis = y

        batch_aa = tf.convert_to_tensor(batch_aa)
        batch_q8 = tf.convert_to_tensor(batch_q8)
        batch_msa = tf.convert_to_tensor(batch_msa)
        batch_seqlen = tf.convert_to_tensor(batch_seqlen)
        batch_dcalphas = tf.convert_to_tensor(batch_dcalphas)
        batch_phis = tf.convert_to_tensor(batch_phis)
        batch_psis = tf.convert_to_tensor(batch_psis)
        return batch_aa, batch_q8, batch_msa, batch_seqlen, batch_dcalphas, batch_phis, batch_psis

    def compute_train_loss(self, x, y):
        batch_aa, batch_q8, batch_msa, batch_seqlen, \
            batch_dcalphas, batch_phis, batch_psis = self.decompress(x, y)


        angle_scale = 180 / np.pi  # convert angles from [-pi, pi] to [-180, 180]

        with tf.GradientTape() as tape:
            y_out = self.model([batch_aa, batch_q8, batch_msa])
            torsion_angles = np.pi * y_out
            phi, psi = torsion_angles[:, :, 0], torsion_angles[:, :, 1]
            phi_scaled, psi_scaled = phi * angle_scale, psi * angle_scale  # from [-pi, pi] to [-180, 180]
            loss_phi_batch = rmsd_torsion_angle(phi_scaled, batch_phis, batch_seqlen)
            loss_psi_batch = rmsd_torsion_angle(psi_scaled, batch_psis, batch_seqlen)
            loss_phi = tf.reduce_mean(loss_phi_batch)
            loss_psi = tf.reduce_mean(loss_psi_batch)

            dist_matrix, coordinates = DistanceMatrix()(torsion_angles)
            loss_drmsd_batch = drmsd_dist_matrix(dist_matrix, batch_dcalphas, batch_seqlen)
            loss_drmsd = tf.reduce_mean(loss_drmsd_batch)  # drmsd metric
            loss_drmsd_normalized = tf.reduce_mean(
                loss_drmsd_batch / tf.sqrt(batch_seqlen))  # longer proteins have larger distance

            # the optimization objective can be 1. loss_drmsd or 2. (loss_phi + loss_psi), or both
            # optimizing distance mattrix (drmsd) and tortion angles separately probably gives better rerults.
            loss_all = loss_drmsd  # distance matrix loss
            # loss_all = loss_phi + loss_psi  # torsion angle loss
            # loss_all = loss_drmsd * dist_loss_weight + (loss_phi + loss_psi) * torsion_loss_weight
            return loss_all, tape

    def train(self, X, Y, steps_per_epoch, test_size=0.2, batch_size=32,
              nb_epochs=100, verbose=True):
        split_idx = X[0].shape[0] * test_size
        train_idx = np.arange(split_idx, X[0].shape[0], dtype=np.int32)
        eval_idx = np.arange(split_idx, dtype=np.int32)

        train_X = [data[train_idx] for data in X]
        train_Y = [data[train_idx] for data in Y]
        test_X = [data[eval_idx] for data in X]
        test_Y = [data[eval_idx] for data in Y]

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(nb_epochs):
            sys.stdout.flush()
            avg_loss = 0.0
            count = 0.0
            with tqdm(total=steps_per_epoch) as pbar:
                for x, y in ProteinModel.gen(train_X, train_Y, steps_per_epoch, batch_size):
                    loss, tape = self.compute_train_loss(x, y)
                    # Compute gradient
                    grads = tape.gradient(loss, model.trainable_weights)  # loss includ both distance matrix and torsion angles
                    # grads = tape.gradient(loss_all, model.trainable_variables)
                    grads, global_norm = tf.clip_by_global_norm(grads, 5.0)

                    # Backprop
                    self.opt.apply_gradients(zip(grads, model.trainable_weights), epoch)

                    avg_loss += loss
                    count += 1
                    pbar.set_postfix(avg_loss=avg_loss / count, epoch=epoch)
                    pbar.update()
            eval_loss = self.evaluate(test_X, test_Y, global_step=epoch, verbose=verbose)
            self.saver.update(epoch=epoch, eval_loss=eval_loss)

    def evaluate(self, test_X, test_Y, global_step, verbose):
        avg_loss = 0.0
        count = 0.0
        for x, y in zip(test_X, test_Y):
            #curr_loss, dp = self.sess.run([self.eval_loss, self.dm_pred], feed_dict=self.gen_feed_dict(x, y))
            loss, tape = self.compute_test_loss(x, y)
            avg_loss += loss
            count += 1

        avg_loss = avg_loss / count
        summary = tf.Summary()
        summary.value.add(tag='eval_loss', simple_value=avg_loss)
        self.summary_writer.add_summary(summary=summary, global_step=global_step)
        if verbose:
            print('Epoch-{} Eval-Loss-{}'.format(global_step, avg_loss))
        return avg_loss

if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    loader = load_loader(mode='train', max_size=200)
    model = ProteinModel(work_dir='./data', per_process_gpu_memory_fraction=0.7,
                         n_words_aa=loader.n_words_aa, n_words_q8=loader.n_words_q8)
    model.train(X=loader.X, Y=loader.Y, steps_per_epoch=2, test_size=0.2, batch_size=32,
                nb_epochs=100, verbose=True)
