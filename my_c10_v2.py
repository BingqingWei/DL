__author__ = 'Bingqing Wei'
from my_lib import *

if __name__ == '__main__':
    trainset, testset = get_c10()
    dataset = MyDataset(batchsize=32, trainset=trainset, testset=testset)
    dataset.set_mode('train')
    input_layer = Input((32, 3072))
    d1_layer = FullyConnected(input_layer, 1024)
    a1_layer = Relu(d1_layer)
    d2_layer = FullyConnected(a1_layer, 256)
    a2_layer = Relu(d2_layer)
    d4_layer = FullyConnected(a2_layer, 10)
    a4_layer = CrossEntropyWithSoftmax(d4_layer)
    model = Model([input_layer, d1_layer,
                   a1_layer, d2_layer,
                   a2_layer, d4_layer,
                   a4_layer],
                  dataset)
    model.train()
