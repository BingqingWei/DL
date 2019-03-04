__author__ = 'Bingqing Wei'

import numpy as np
import scipy
from scipy.special import comb
import os
import pickle
from tensorflow.keras.preprocessing import text, sequence
from sklearn.utils import shuffle
import gc
import math

#actually used only 20 amino acids: amino_acids_letters = 'HSEVAKYRCNLWPDIMFGQT-'

# WARNING: DEPRECATED
amino_acids_letters = 'RHKDESTNQCUGPAVILMFYW'
aa2int = dict([(x, i) for i, x in enumerate(amino_acids_letters)])
int2aa = dict([(i, x) for i, x in enumerate(amino_acids_letters)])

secondary_structure_letters = 'GHITEBS-'
ss2int = dict([(x, i) for i, x in enumerate(secondary_structure_letters)])
int2ss = dict([(i, x) for i, x in enumerate(secondary_structure_letters)])

def seq2ngrams(seqs, ngrams):
    return np.array([[seq[i : i + ngrams] for i in range(len(seq))] for seq in seqs])

class BatchDataLoader:
    """
    X: amino_acids, secondary_structure, msa_features
    Y: distance_matrix, torsion_angles
    """

    def __init__(self, data_dir, mode,
                 max_size=None, max_seq_len=384, min_seq_len=10,
                 ngrams=1, load_dir=None):
        assert mode in ['test', 'train']
        self.data_dir = data_dir
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.max_size = max_size

        self.ngrams = ngrams
        if load_dir is None:
            if mode == 'test':
                with open(os.path.join(data_dir, self.mode + '_tokenizers.pkl'), 'rb') as f:
                    self.tokenizer_aa, self.tokenizer_q8 = pickle.load(f)
                self.files = [os.path.join(data_dir, 'test.pkl')]
            else:
                self.files = [os.path.join(data_dir, 'train_fold_{}.pkl'.format(i)) for i in range(1, 11)]
            self.X = []
            self.Y = []
            self.tokenizer_aa = None
            self.tokenizer_q8 = None
            self.load_all(data_dir)
            self.n_words_aa = None
            self.n_words_q8 = None
            self.aa_total = int(comb(N=len(amino_acids_letters), k=ngrams))
            self.ss_total = int(comb(N=len(secondary_structure_letters), k=ngrams))
        else:
            self.load(load_dir)

        if self.max_size is not None:
            self.X = [l[:self.max_size] for l in self.X]
            self.Y = [l[:self.max_size] for l in self.Y]

    def get_data(self):
        return self.X, self.Y

    def gen(self, steps):
        idx_shuffle = np.random.permutation(self.X[0].shape[0])
        for i in range(steps):
            x_train = [data[idx_shuffle] for data in self.X]
            y_train = [data[idx_shuffle] for data in self.Y]
            yield x_train, y_train

    def load(self, load_dir):
        with open(os.path.join(load_dir, self.mode + '_tokenizers.pkl'), 'rb') as f:
            self.tokenizer_aa, self.tokenizer_q8 = pickle.load(f)
        with open(os.path.join(load_dir, self.mode + '_X.pkl'), 'rb') as f:
            self.X = pickle.load(f)
        with open(os.path.join(load_dir, self.mode + '_Y.pkl'), 'rb') as f:
            self.Y = pickle.load(f)
        self.n_words_aa = len(self.tokenizer_aa.word_index) + 1
        self.n_words_q8 = len(self.tokenizer_q8.word_index) + 1

    def saveTo(self, path):
        with open(os.path.join(path, self.mode + '_X.pkl'), 'wb') as f:
            pickle.dump(self.X, f)
        with open(os.path.join(path, self.mode + '_Y.pkl'), 'wb') as f:
            pickle.dump(self.Y, f)
        with open(os.path.join(path, self.mode + '_tokenizers.pkl'), 'wb') as f:
            li = [self.tokenizer_aa, self.tokenizer_q8]
            pickle.dump(li, f)

    def load_all(self, data_dir):
        """
        :train:
        tokenizers: tokenizer_aa, tokenizer_q8,
        X: [train_aa_grams, train_q8_grams, msas_padded, length_seq],
        Y: [dcalphas_pad, train_target_phis, train_target_psis]

        :test:
        Not Implemented
        """
        if self.mode == 'train':
            # The number of words and tags to be passed as parameters to the model
            self.tokenizer_aa, self.tokenizer_q8, self.X, self.Y = self.load_all_train()
            self.n_words_aa = len(self.tokenizer_aa.word_index) + 1
            self.n_words_q8 = len(self.tokenizer_q8.word_index) + 1
        else:
            with open(os.path.join(data_dir, 'train' + '_tokenizers.pkl'), 'rb') as f:
                self.tokenizer_aa, self.tokenizer_q8 = pickle.load(f)
            self.X = self.load_all_test()

    def load_all_test(self):
        data_fields = [[] for i in range(6)]
        for fpath in self.files:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
                # amino acids, secondary structure, msa features
                for j in range(len(data_fields)):
                    data_fields[j].append(data[j])
        for j in range(len(data_fields)):
            data_fields[j] = np.concatenate(data_fields[j])
        indices, pdbs, length_seq, aa_seq, q8_seq, msas = data_fields
        msas = [np.stack(x).transpose().astype(np.float32) for x in msas]

        # processing msa
        msa_dim = msas[0].shape[1]
        msas_padded = np.zeros([len(msas), self.max_seq_len, msa_dim], dtype=np.float32)
        for i in range(len(msas)):
            msas_padded[i, :msas[i].shape[0], :] = msas[i]
        del msas, data_fields

        train_aa_grams = seq2ngrams(aa_seq, ngrams=self.ngrams)
        train_q8_grams = seq2ngrams(q8_seq, ngrams=self.ngrams)

        train_aa_grams = self.tokenizer_aa.texts_to_sequences(train_aa_grams)
        train_aa_grams = sequence.pad_sequences(train_aa_grams, maxlen=self.max_seq_len,
                                                  padding='post', truncating='post')
        train_q8_grams = self.tokenizer_q8.texts_to_sequences(train_q8_grams)
        train_q8_grams = sequence.pad_sequences(train_q8_grams, maxlen=self.max_seq_len,
                                                padding='post', truncating='post')
        return train_aa_grams, train_q8_grams, msas_padded, length_seq

    def load_all_train(self):
        data_fields = [[] for i in range(9)]
        for fpath in self.files:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
                # amino acids, secondary structure, msa features
                for j in range(len(data_fields)):
                    data_fields[j].append(data[j])
        for j in range(len(data_fields)):
            data_fields[j] = np.concatenate(data_fields[j])

        indices, pdbs, length_seq, pdf_aas, q8s, dcalphas, psis, phis, msas = data_fields
        filter_mask = (length_seq < self.max_seq_len) & (length_seq > self.min_seq_len)
        for i in range(len(data_fields)):
            data_fields[i] = data_fields[i][filter_mask]

        indices, pdbs, length_seq, aa_seq, q8_seq, dcalphas, psis, phis, msas = data_fields
        msas = [np.stack(x).transpose().astype(np.float32) for x in msas]
        phis = [np.array(x) for x in phis]
        psis = [np.array(x) for x in psis]

        # processing distance matrix
        dcalphas_pad = np.zeros((len(dcalphas), self.max_seq_len, self.max_seq_len), dtype=np.float32)
        for i in range(len(dcalphas)):
            length = dcalphas[i].shape[0]
            dcalphas_pad[i, :length, :length] = dcalphas[i]
        del dcalphas

        # processing torsion angles
        train_target_phis = np.zeros([len(phis), self.max_seq_len], dtype=np.float32)
        for i in range(len(phis)):
            train_target_phis[i, :phis[i].shape[0]] = phis[i]
        train_target_psis = np.zeros([len(psis), self.max_seq_len], dtype=np.float32)
        for i in range(len(psis)):
            train_target_psis[i, :psis[i].shape[0]] = psis[i]
        del phis, psis

        # processing msa
        msa_dim = msas[0].shape[1]
        msas_padded = np.zeros([len(msas), self.max_seq_len, msa_dim], dtype=np.float32)
        for i in range(len(msas)):
            msas_padded[i, :msas[i].shape[0], :] = msas[i]
        del msas, data_fields

        train_aa_grams = seq2ngrams(aa_seq, ngrams=self.ngrams)
        train_q8_grams = seq2ngrams(q8_seq, ngrams=self.ngrams)

        tokenizer_aa = text.Tokenizer()
        tokenizer_aa.fit_on_texts(train_aa_grams)
        tokenizer_q8 = text.Tokenizer()
        tokenizer_q8.fit_on_texts(train_q8_grams)

        train_aa_grams = tokenizer_aa.texts_to_sequences(train_aa_grams)
        train_aa_grams = sequence.pad_sequences(train_aa_grams, maxlen=self.max_seq_len,
                                                  padding='post', truncating='post')
        train_q8_grams = tokenizer_q8.texts_to_sequences(train_q8_grams)
        train_q8_grams = sequence.pad_sequences(train_q8_grams, maxlen=self.max_seq_len,
                                                padding='post', truncating='post')
        return tokenizer_aa, tokenizer_q8, [train_aa_grams, train_q8_grams, msas_padded, length_seq],\
               [dcalphas_pad, train_target_phis, train_target_psis]



def save_data(mode='train', ngrams=1, work_dir=os.path.join('..', 'data'), max_size=None):
    loader = BatchDataLoader(data_dir=work_dir, mode=mode, ngrams=ngrams, max_size=max_size)
    loader.saveTo(work_dir)

def load_loader(mode='train', work_dir=os.path.join('..', 'data'), max_size=None):
    loader = BatchDataLoader(data_dir=None, mode=mode, load_dir=work_dir, max_size=max_size)
    return loader

if __name__ == '__main__':
    save_data(mode='test')
    #loader = load_loader()
