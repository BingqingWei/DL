__author__ = 'Bingqing Wei'
import numpy as np
from my_loader import *
import torch
from collections import Counter

def init_weight(shape):
    norm = np.sqrt(2.0 / (np.sum(shape)))
    return np.random.randn(shape[0], shape[1]) * norm

class Layer:
    def __init__(self, prev, input_shape):
        assert len(input_shape) == 2
        self.prev = prev
        self.input_shape = input_shape
        self.grads = {} # key: parameter, value: (batch_size, parameter shape)
        self.mode = 'train'

    def backward(self, dy):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def get_output_shape(self):
        raise NotImplementedError()

    def set_mode(self, mode='train'):
        self.mode = mode


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

class Tanh(Layer):
    def __init__(self, prev):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(Tanh, self).__init__(prev, input_shape)
        self.gz_x = None
        self.lz_x = None

    def backward(self, dy):
        if self.gz_x is None or self.lz_x is None:
            raise ValueError('forward not called before backward')
        return 4 / np.square(self.gz_x + self.lz_x)

    def forward(self, x):
        self.gz_x = np.exp(x)
        self.lz_x = np.exp(-x)
        return (self.gz_x - self.lz_x) / (self.gz_x + self.lz_x)


    def get_output_shape(self):
        return self.input_shape


class Sigmoid(Layer):
    def __init__(self, prev):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(Sigmoid, self).__init__(prev, input_shape)
        self.out = None

    def backward(self, dy):
        if self.out is None:
            raise ValueError('forward not called before backward')
        return dy * self.out * (1 - self.out)

    def forward(self, x):
        x = x.copy()
        gz = x >= 0
        x[gz] = 1 / (1 + np.exp(-x[gz]))
        lz = x < 0
        x[lz] = x[lz] / (1 + x[lz])
        self.out = x
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

class DropOut(Layer):
    def __init__(self, prev, rate):
        '''
        :param prev: previous Layer
        :param rate: zero out rate
        '''
        assert isinstance(prev, Layer)
        assert 0 <= rate <= 1
        input_shape = prev.get_output_shape()
        super(DropOut, self).__init__(prev, input_shape)
        self.rate = rate
        self.mask = None

    def forward(self, x):
        if self.mode == 'train':
            self.mask = np.random.choice([0, 1], size=self.input_shape, p=[self.rate, 1 - self.rate])
            return self.mask * x
        else:
            return (1 - self.rate) * x

    def backward(self, dy):
        if self.mask is None:
            raise ValueError('forward not called before backward')
        return self.mask * dy

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

    def optimize(self, lr=0.01):
        self.backward()
        for layer in reversed(self.layers):
            for w in layer.grads.keys():
                update = lr * layer.grads[w]
                x = getattr(layer, w)
                x -= update

    def eval(self, result, y):
        x1 = np.argmax(result, axis=1)
        x2 = np.argmax(y, axis=1)
        corr = np.sum(x1 == x2)
        #corr = np.sum(np.argmax(result, axis=1) == np.argmax(y, axis=1))
        return corr / result.shape[0]


    def print(self, counter, epoch_nb, epoch_end=False, freq=100, mode='train'):
        if epoch_end:
            print(mode.upper() + ': Epoch {} counts {}/ CR:{}, Loss:{}'.format(epoch_nb, counter['count'],
                                                                              counter['cr']/counter['count'],
                                                                              counter['loss']/counter['count']))
            print('#' * 30)
        else:
            if counter['count'] % freq == 1:
                print('Epoch {} counts {}/ CR:{}, Loss:{}'.format(epoch_nb, counter['count'],
                                                                  counter['cr']/counter['count'],
                                                                  counter['loss']/counter['count']))

    def set_mode(self, mode='train'):
        assert mode in ['train', 'test']
        for l in self.layers:
            l.set_mode(mode)
        self.dataset.set_mode(mode)


    def train(self, epoch=100):
        def reset(counter):
            for k in counter:
                counter[k] = 0

        counter = Counter()
        counter['cr'] = 0
        counter['loss'] = 0
        counter['count'] = 0
        for e in range(epoch):
            self.set_mode('train')
            reset(counter)
            for i in range(len(self.dataset)):
                x, y = self.dataset[i]
                r, loss = self.forward(x, y)
                self.optimize()
                cr = self.eval(r, y)
                counter['count'] += 1
                counter['cr'] += cr
                counter['loss'] += loss
                self.print(counter, epoch_nb=e, epoch_end=False, freq=100, mode='train')

            self.print(counter, epoch_nb=e, epoch_end=True, freq=100, mode='train')
            reset(counter)
            self.set_mode('test')
            for i in range(len(self.dataset)):
                x, y = self.dataset[i]
                r, loss = self.forward(x, y)
                cr = self.eval(r, y)
                counter['count'] += 1
                counter['cr'] += cr
                counter['loss'] += loss
                self.print(counter, epoch_nb=e, epoch_end=False, freq=100, mode='test')
            self.print(counter, epoch_nb=e, epoch_end=True, freq=100, mode='test')
