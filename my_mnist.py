__author__ = 'Bingqing Wei'
import numpy as np
from my_loader import *
import torch

def init_weight(shape):
    norm = np.sqrt(2.0 / (np.sum(shape)))
    return np.random.randn(shape[0], shape[1]) * norm

class Layer:
    def __init__(self, prev, input_shape):
        assert len(input_shape) == 2
        self.prev = prev
        self.input_shape = input_shape
        self.grads = {} # key: parameter, value: (batch_size, parameter shape)

    def backward(self, dy):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def get_output_shape(self):
        raise NotImplementedError()

class FullyConnected(Layer):
    def __init__(self, prev, hidden_nb):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(FullyConnected, self).__init__(prev, input_shape)
        self.W = init_weight((input_shape[-1], hidden_nb))
        self.bias = np.zeros(hidden_nb) + 0.01
        self.x = None

    def backward(self, dy):
        if self.x is None:
            raise ValueError('forward not called before backward')
        self.grads['W'] = np.matmul(np.transpose(self.x), dy) / dy.shape[0]
        self.grads['bias'] = np.average(dy, 0)
        return np.matmul(dy, np.transpose(self.W))

    def forward(self, x):
        self.x = x
        return np.matmul(x, self.W) + self.bias

    def get_output_shape(self):
        return self.input_shape[0], self.W.shape[-1]

class Input(Layer):
    def __init__(self, input_shape):
        super(Input, self).__init__(None, input_shape)

    def backward(self, dy):
        pass

    def forward(self, x):
        return x

    def get_output_shape(self):
        return self.input_shape

class Relu(Layer):
    def __init__(self, prev):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(Relu, self).__init__(prev, input_shape)

    def backward(self, dy):
        return dy

    def forward(self, x):
        x[x < 0] = 0
        return x

    def get_output_shape(self):
        return self.input_shape

class CrossEntropyWithSoftmax(Layer):
    def __init__(self, prev):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(CrossEntropyWithSoftmax, self).__init__(prev, input_shape)
        self.sft = None
        self.y = None

    def backward(self, dy):
        if self.sft is None or self.y is None:
            raise ValueError('forward not called before backward')
        return dy * (self.sft - self.y)

    def forward(self, x):
        shift = x - np.expand_dims(np.max(x, axis=1), axis=1)
        exps = np.exp(shift)
        self.sft = np.apply_along_axis(lambda z: z / np.sum(z), 1, exps)
        return self.sft

    def loss(self, y):
        if self.sft is None:
            raise ValueError('forward not called before computing loss')
        self.y = y
        return -np.sum(y * np.log(self.sft)) / y.shape[0]

    def get_output_shape(self):
        return self.input_shape[0], 1

class Model:
    def __init__(self, layers, dataset):
        self.layers = layers
        self.dataset = dataset

    def forward(self, x, y):
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        loss = self.layers[-1].loss(y)
        return x, loss

    def backward(self):
        dy = 1.0
        for i in range(len(self.layers) - 1, -1, -1):
            dy = self.layers[i].backward(dy)

    def optimize(self, learnin_rate=0.01):
        self.backward()
        for layer in reversed(self.layers):
            for w in layer.grads.keys():
                update = learnin_rate * layer.grads[w]
                x = getattr(layer, w)
                x -= update

    def eval(self, result, y):
        x1 = np.argmax(result, axis=1)
        x2 = np.argmax(y, axis=1)
        corr = np.sum(x1 == x2)
        #corr = np.sum(np.argmax(result, axis=1) == np.argmax(y, axis=1))
        return corr / result.shape[0]

    def train(self, epoch=100):
        print('Dataset size {} batches'.format(len(self.dataset)))
        from collections import Counter
        counter = Counter()
        for e in range(epoch):
            counter['cr'] = 0
            counter['loss'] = 0
            counter['count'] = 0
            for x, y in self.dataset:
                r, loss = self.forward(x, y)
                self.optimize()
                cr = self.eval(r, y)
                if counter['count'] % 100 == 1:
                    print('Epoch {} counts {}: CR-{}, Loss-{}'.format(e, counter['count'],
                                                                      counter['cr']/counter['count'],
                                                                      counter['loss']/counter['count']))
                counter['count'] += 1
                counter['cr'] += cr
                counter['loss'] += loss
            print('#' * 30)
            print('Epoch {} counts {}: CR-{}, Loss-{}'.format(e, counter['count'],
                                                              counter['cr']/counter['count'],
                                                              counter['loss']/counter['count']))


if __name__ == '__main__':
    trainset, testset = get_mnist()
    dataset = MyDataset(batchsize=4, trainset=trainset, testset=testset)
    dataset.set_mode('train')
    input_layer = Input((4, 784))
    d1_layer = FullyConnected(input_layer, 1024)
    a1_layer = Relu(d1_layer)
    d2_layer = FullyConnected(a1_layer, 1024)
    a2_layer = Relu(d2_layer)
    d3_layer = FullyConnected(a2_layer, 10)
    a4_layer = CrossEntropyWithSoftmax(d3_layer)
    model = Model([input_layer, d1_layer, a1_layer, d2_layer, a2_layer,
                   d3_layer, a4_layer],
                  dataset)
    model.train()


