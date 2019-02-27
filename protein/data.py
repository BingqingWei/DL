__author__ = 'Bingqing Wei'

import numpy as np
import scipy
from scipy.special import comb
import os
import pickle

#actually used only 20 amino acids: amino_acids_letters = 'HSEVAKYRCNLWPDIMFGQT-'
amino_acids_letters = 'RHKDESTNQCUGPAVILMFYW'
aa2int = dict([(x, i) for i, x in enumerate(amino_acids_letters)])

secondary_structure_letters = 'GHITEBS-'
ss2int = dict([(x, i) for i, x in enumerate(secondary_structure_letters)])

"""
DataLoader: 
the data loader for protein tertiary structure prediction

The batch size will always be 1.
"""

class DataLoader:
    """
    X: amino_acids, secondary_structure, msa_features
    Y: distance_matrix, torsion_angles
    """

    def __init__(self, data_dir, mode, ngrams=1, load_dir=None):
        assert mode in ['test', 'train']
        self.data_dir = data_dir
        self.mode = mode
        self.obj_index = -1

        self.ngrams = ngrams
        if load_dir is None:
            if mode == 'test':
                self.files = [os.path.join(data_dir, 'test.pkl')]
            else:
                self.files = [os.path.join(data_dir, 'train_fold_{}.pkl'.format(i)) for i in range(1, 11)]
            self.X = []
            self.Y = []
            self.load_all()
            self.aa_total = int(comb(N=len(amino_acids_letters), k=ngrams))
            self.ss_total = int(comb(N=len(secondary_structure_letters), k=ngrams))
        else:
            self.load(load_dir)

    def get_data(self):
        return self.X, self.Y

    def load(self, load_dir):
        with open(os.path.join(load_dir, self.mode + '_X.pkl'), 'rb') as f:
            self.X = pickle.load(f)
        with open(os.path.join(load_dir, self.mode + '_Y.pkl'), 'rb') as f:
            self.Y = pickle.load(f)

    def saveTo(self, path):
        with open(os.path.join(path, self.mode + '_X.pkl'), 'wb') as f:
            pickle.dump(self.X, f)
        with open(os.path.join(path, self.mode + '_Y.pkl'), 'wb') as f:
            pickle.dump(self.Y, f)


    def load_all(self):
        for fpath in self.files:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
                # amino acids, secondary structure, msa features
                for i in range(len(data[0])):
                    self.X.append(self.preprocess_X(data[3][i], data[4][i], data[8][i]))
                    self.Y.append(self.preprocess_Y(data[5][i], data[6][i], data[7][i]))

    def get_loader(self):
        while True:
            self.obj_index = (self.obj_index + 1) % len(self.X)
            yield np.expand_dims(self.X[self.obj_index], axis=0), \
                  np.expand_dims(self.Y[self.obj_index], axis=0)

    def preprocess_X(self, amino_acids, secondary_structure, msa_feature):
        amino_acids = list(map(lambda x: aa2int[x], amino_acids))
        secondary_structure = list(map(lambda x: ss2int[x], secondary_structure))

        # shape = (N choose k, ngram)
        amino_acids = np.asarray(list(zip(*[amino_acids[i:] for i in range(self.ngrams)])), dtype=np.int32)
        secondary_structure = np.asarray(list(zip(*[secondary_structure[i:] for i in range(self.ngrams)])), dtype=np.int32)
        if self.ngrams == 1:
            amino_acids = np.squeeze(amino_acids, axis=-1)
            secondary_structure = np.squeeze(secondary_structure, axis=-1)

        # shape = (N choose k, ngram, 21)
        msa_feature = np.transpose(np.asarray(msa_feature, dtype=np.float32)) #resulting in n * 21 matrix
        msa_feature = np.asarray(list(zip(*[msa_feature[i:] for i in range(self.ngrams)])))
        if self.ngrams == 1:
            msa_feature = np.squeeze(msa_feature, axis=1)
        return amino_acids, secondary_structure, msa_feature


    def preprocess_Y(self, distance_matrix, torsion_psi, torsion_phi):
        """
        :param distance_matrix: (n, n)
        :param torsion_psi: (n,)
        :param torsion_phi: (n,)
        :return: distance_matrix(vector): (n * (n - 1) / 2,), torsion_psi: (n,), torsion_phi: (n,)
        """
        #distance_matrix = np.concatenate([distance_matrix[i][i + 1:] for i in range(distance_matrix.shape[0] - 1)])
        distance_matrix = np.asarray(distance_matrix)
        torsion_psi = np.asarray(torsion_psi) / 360.0
        torsion_phi = np.asarray(torsion_phi) / 360.0
        return distance_matrix, np.transpose(np.stack([torsion_psi, torsion_phi], axis=0))

    @staticmethod
    def reverse_Y_concat(p_matrix, p_torsion):
        p_psi = p_torsion[:][0]
        p_phi = p_torsion[:][1]
        return DataLoader.reverse_Y(p_matrix, p_psi, p_phi)

    @staticmethod
    def reverse_Y(p_matrix, p_psi, p_phi):
        # reverse the predicted distance_matrix, torsion_psi, torsion_phi
        matrix = np.zeros(shape=(p_psi.shape[0], p_psi.shape[0]))
        count = 0
        for i in range(p_psi.shape[0]):
            for j in range(i + 1, p_psi.shape[0]):
                matrix[i][j] = p_matrix[count]
                matrix[j][i] = p_matrix[count]
                count += 1

        p_psi *= 360.0
        p_phi *= 360.0
        return p_matrix, p_psi, p_phi

def save_data(mode='train', ngrams=1, work_dir=os.path.join('.', 'data')):
    loader = DataLoader(data_dir=os.path.join('.', 'data'), mode=mode, ngrams=ngrams)
    loader.saveTo(work_dir)

def load_loader(mode='train', work_dir=os.path.join('.', 'data')):
    loader = DataLoader(data_dir=None, mode=mode, load_dir=work_dir)
    return loader

if __name__ == '__main__':
    save_data()
    #loader = load_loader()
