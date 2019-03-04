__author__ = 'Bingqing Wei'
import sys
import numpy as np
from keras.layers import Concatenate, Embedding, LeakyReLU, LSTM, ReLU, TimeDistributed, \
    Bidirectional, GRU, Dense
from keras import activations
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle
import keras.backend as K

from model3_batch.data import *
from model3_batch.utils_tf import *

class ProteinModel:
    def __init__(self, work_dir, n_words_aa, n_words_q8,
                 per_process_gpu_memory_fraction=0.7):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = per_process_gpu_memory_fraction
        self.sess = tf.Session(config=config)

        self.seq_len_tensor, self.mask_ta_tensor, self.mask_dist_tensor, \
            self.msa_tensor, self.amino_tensor, self.secondary_tensor, \
            self.target_distance, self.target_torsion_psi, self.target_torsion_phi, \
            self.train_op, self.train_loss, self.eval_loss, \
            self.dm_pred, self.ta_pred = self.build_model(n_words_aa, n_words_q8)

        self.graph_saver = CkptSaver(work_dir=work_dir, sess=self.sess)

        self.work_dir = work_dir
        self.summary_writer = tf.summary.FileWriter(logdir=work_dir, graph=self.sess.graph)

    def get_tensors(self):
        return self.msa_tensor, self.amino_tensor, self.secondary_tensor, \
            self.target_distance, self.train_op

    def build_model(self, n_words_aa, n_words_q8):
        """
        :return msa_tensor, amino_tensor, secondary_tensor, \
               target_distance, target_torsion, train_op, train_loss, eval_loss
        """
        msa_tensor = tf.placeholder(shape=(None, None, 21), dtype=tf.float32)
        amino_tensor = tf.placeholder(shape=(None, None), dtype=tf.int32)
        secondary_tensor = tf.placeholder(shape=(None, None), dtype=tf.int32)

        # generated using length_seq
        seq_len_tensor = tf.placeholder(shape=(None,), dtype=tf.float32)
        mask_ta_tensor = tf.placeholder(shape=(None, None), dtype=tf.float32)
        mask_dist_tensor = tf.placeholder(shape=(None, None, None), dtype=tf.float32)

        target_distance = tf.placeholder(shape=(None, None, None), dtype=tf.float32)
        target_torsion_psi = tf.placeholder(shape=(None, None), dtype=tf.float32)
        target_torsion_phi = tf.placeholder(shape=(None, None), dtype=tf.float32)

        amino_embed = Embedding(input_dim=n_words_aa, output_dim=128)(amino_tensor)
        secondary_embed = Embedding(input_dim=n_words_q8, output_dim=64)(secondary_tensor)

        x = Concatenate(axis=-1)([msa_tensor, secondary_embed, amino_embed])
        x = Bidirectional(GRU(units=64, return_sequences=True,
                              recurrent_dropout=0.1, recurrent_activation='relu'))(x)
        ta_pred = TimeDistributed(Dense(units=3, activation=activations.tanh))(x)

        phi = ta_pred[:, :, 0]
        psi = ta_pred[:, :, 1]

        loss_phi = tf.reduce_mean(rmsd_torsion_angle(phi * 180.0, target_torsion_phi, seq_len_tensor, mask_ta_tensor))
        loss_psi = tf.reduce_mean(rmsd_torsion_angle(psi * 180.0, target_torsion_psi, seq_len_tensor, mask_ta_tensor))

        dm_pred, _ = get_distance_matrix(ta_pred * np.pi)
        loss_drmsd = drmsd_dist_matrix(dm_pred, target_distance, seq_len_tensor, mask_dist_tensor)

        train_loss = tf.reduce_mean(loss_drmsd) + 1.0 / 50 * (loss_phi + loss_psi)
        eval_loss = tf.sqrt(tf.losses.mean_squared_error(labels=target_distance, predictions=dm_pred))


        eval_loss += 0.02 * tf.sqrt(tf.losses.mean_squared_error(labels=target_torsion_phi, predictions=ta_pred[:, :, 0]))
        eval_loss += 0.02 * tf.sqrt(tf.losses.mean_squared_error(labels=target_torsion_psi, predictions=ta_pred[:, :, 1]))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(train_loss, tvars), 5.0)
        train_op = tf.train.RMSPropOptimizer(learning_rate=0.0005).apply_gradients(zip(grads, tvars))
        return seq_len_tensor, mask_ta_tensor, mask_dist_tensor, \
               msa_tensor, amino_tensor, secondary_tensor, \
               target_distance, target_torsion_psi, target_torsion_phi, \
               train_op, train_loss, eval_loss, \
               dm_pred, ta_pred

    def gen_feed_dict(self, x, y):
        dist_mask = np.zeros(shape=y[0].shape, dtype=np.float32)
        for i, length in enumerate(x[3]):
            dist_mask[i, :length, :length] = 1

        dist_ta = np.zeros(shape=y[1].shape, dtype=np.int32)
        for i, length in enumerate(x[3]):
            dist_ta[i, :length] = 1.0


        feed_dict = {
            self.msa_tensor: x[2],
            self.secondary_tensor: x[1],
            self.amino_tensor: x[0],
            self.seq_len_tensor: x[3],

            self.target_distance: y[0],
            self.target_torsion_psi: y[1],
            self.target_torsion_phi: y[2],
            self.mask_ta_tensor: dist_ta,
            self.mask_dist_tensor: dist_mask
        }
        return feed_dict

    @staticmethod
    def gen(X, Y, steps, batch_size):
        idx_shuffle = np.random.permutation(X[0].shape[0])
        if steps is None: steps = int(math.floor(X[0].shape[0] / batch_size))

        for i in range(steps):
            if (i + 1) * batch_size >= len(idx_shuffle): break
            idx_batch = idx_shuffle[i * batch_size : (i+1) * batch_size]
            x_train = [data[idx_batch] for data in X]
            y_train = [data[idx_batch] for data in Y]
            yield x_train, y_train

    def train(self, X, Y, batch_size=32,
              test_size=0.2, nb_epochs=100, verbose=True):

        split_idx = X[0].shape[0] * test_size
        train_idx = np.arange(split_idx, X[0].shape[0], dtype=np.int32)
        eval_idx = np.arange(split_idx, dtype=np.int32)
        train_X = [data[train_idx] for data in X]
        train_Y = [data[train_idx] for data in Y]
        test_X = [data[eval_idx] for data in X]
        test_Y = [data[eval_idx] for data in Y]

        self.sess.run(tf.global_variables_initializer())

        steps_per_epoch = int(math.floor(X[0].shape[0] / batch_size))
        for epoch in range(nb_epochs):
            sys.stdout.flush()
            avg_loss = 0.0
            count = 0.0
            with tqdm(total=steps_per_epoch) as pbar:
                for x, y in ProteinModel.gen(train_X, train_Y, steps_per_epoch, batch_size):
                    loss, _ = self.sess.run([self.train_loss, self.train_op],
                                            feed_dict=self.gen_feed_dict(x, y))
                    avg_loss += loss
                    count += 1
                    pbar.set_postfix(avg_loss=avg_loss / count, epoch=epoch)
                    pbar.update()
            eval_loss = self.evaluate(test_X, test_Y, global_step=epoch, verbose=verbose)
            self.graph_saver.update(epoch=epoch, eval_loss=eval_loss)

    def evaluate(self, test_X, test_Y, global_step, verbose, batch_size=8):
        avg_loss = 0.0
        count = 0.0
        for x, y in ProteinModel.gen(test_X, test_Y, None, batch_size):
            #curr_loss, dp = self.sess.run([self.eval_loss, self.dm_pred], feed_dict=self.gen_feed_dict(x, y))
            curr_loss = self.sess.run(self.eval_loss, feed_dict=self.gen_feed_dict(x, y))
            avg_loss += curr_loss
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
    loader = load_loader(mode='train', max_size=1000)
    model = ProteinModel(work_dir='./data', per_process_gpu_memory_fraction=0.7,
                         n_words_aa=loader.n_words_aa, n_words_q8=loader.n_words_q8)
    model.train(X=loader.X, Y=loader.Y, test_size=0.2, batch_size=32,
                nb_epochs=100, verbose=True)