__author__ = 'Bingqing Wei'
import pickle
import os

def checkMSA():
    with open(os.path.join('..', 'data', 'test.pkl'), 'rb') as f:
        data = pickle.load(f)
    msa = data[-1]
    pass

def checkAminoAcids():
    acids = set()
    out = []

    for i in range(1, 11):
        with open(os.path.join('..', 'data', 'train_fold_{}.pkl'.format(i)), 'rb') as f:
            data = pickle.load(f)

        for j in range(len(data[3])):
            s = set(data[3][j])
            acids.update(s)
            out.append(len(s))

    with open(os.path.join('..', 'data', 'test.pkl'), 'rb') as f:
        data = pickle.load(f)
        for j in range(len(data[3])):
            s = set(data[3][j])
            acids.update(s)
            out.append(len(s))

    print('Total number of samples:', len(out))
    print('#####size of amino acids#####')
    print(out)
    print('####amino acids####')
    print('Total size of amino acids:', len(acids))
    print(acids)

if __name__ == '__main__':
    checkAminoAcids()
