__author__ = 'Bingqing Wei'
from my_lib import *

if __name__ == '__main__':
    trainset, testset = get_c10()
    dataset = MyDataset(batchsize=32, trainset=trainset, testset=testset)
    dataset.set_mode('train')
    input_layer = Input((32, 3072))
    d1_layer = FullyConnected(input_layer, 2048)
    a1_layer = Relu(d1_layer)
    c1_layer = DropOut(a1_layer, 0.5)

    d2_layer = FullyConnected(c1_layer, 1024)
    a2_layer = Relu(d2_layer)
    c2_layer = DropOut(a2_layer, 0.5)

    d3_layer = FullyConnected(c2_layer, 1024)
    a3_layer = Relu(d3_layer)
    c3_layer = DropOut(a3_layer, 0.5)

    d4_layer = FullyConnected(c3_layer, 256)
    a4_layer = Relu(d4_layer)
    c4_layer = DropOut(a4_layer, 0.5)

    d9_layer = FullyConnected(c4_layer, 10)

    a9_layer = CrossEntropyWithSoftmax(d9_layer)
    model = Model([input_layer, d1_layer, a1_layer, c1_layer,
                   d2_layer, a2_layer, c2_layer,
                   d3_layer, a3_layer, c3_layer,
                   d4_layer, a4_layer, c4_layer,
                   d9_layer, a9_layer],
                  dataset)
    model.train()
