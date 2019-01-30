__author__ = 'Bingqing Wei'
import numpy as np
from loader import C10Dataset

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
        self.W = np.random.rand(input_shape[-1], hidden_nb)
        self.bias = np.random.rand(hidden_nb)
        self.x = None

    def backward(self, dy):
        if self.x is None:
            raise ValueError('forward not called before backward')
        self.grads[self.W] = np.matmul(dy, self.x)
        self.grads[self.bias] = dy
        return self.grads[self.W]

    def forward(self, x):
        self.x = x
        out = self.prev.forward(x)
        return np.matmul(out, self.W) + self.bias

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
        self.sign = None

    def backward(self, dy):
        if self.sign is None:
            raise ValueError('forward not called before backward')
        return self.sign

    def forward(self, x):
        self.sign = np.apply_along_axis(lambda x: 1 if x >= 0 else 0, 1, x)
        self.sign.reshape((x.shape[0], 1))
        return np.apply_along_axis(lambda z: z if z >= 0 else 0, 1, x)

    def get_output_shape(self):
        return self.input_shape

class CrossEntropy(Layer):
    def __init__(self, prev):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(CrossEntropy, self).__init__(prev, input_shape)
        self.x = None
        self.y = None

    def backward(self, dy):
        if self.y is None or self.x is None:
            raise ValueError('forward not called before backward')
        # should be - ti / yi but dont know how to deal with divide by 0

    def forward(self, x, y):
        self.x = x
        self.y = y
        return -np.sum(y * np.log(x)) / x.shape[1]

    def get_output_shape(self):
        return self.input_shape[0], 1

class Softmax(Layer):
    def __init__(self, prev):
        assert isinstance(prev, Layer)
        input_shape = prev.get_output_shape()
        super(Softmax, self).__init__(prev, input_shape)
        self.x = None

    def backward(self, dy):
        #TODO don't know how to write it
        pass

    def forward(self, x):
        shift = x - np.max(x, axis=1)
        exps = np.exp(shift)
        return exps / np.sum(exps, axis=1)

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
        shift = x - np.max(x, axis=1)
        exps = np.exp(shift)
        self.sft = exps / np.sum(exps, axis=1)

    def loss(self, y):
        if self.sft is None:
            raise ValueError('forward not called before computing loss')
        self.y = y
        return -np.sum(y * np.log(self.sft)) / x.shape[1]

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
        for layer in self.layers:
            for w in layer.grads.keys():
                update = learnin_rate * np.average(layer.grads[w], axis=0)
                w -= update

    def eval(self, result, y):
        corr = np.sum(np.argmax(result, axis=1) == np.argmax(y, axis=1))
        return corr / result.shape[0]

    def train(self, epoch=100):
        for e in range(epoch):
            count = 0
            cumulative_cr = 0
            for x, y in self.dataset:
                r, loss = self.forward(x, y)
                self.optimize()
                cr = self.eval(r, y)
                print('Epoch {}: {}'.format(e, cr))
                count += 1
                cumulative_cr += cr
            print('#' * 30)
            print('Epoch {} cumulative correct rate: {}'.format(e, cumulative_cr / count))


if __name__ == '__main__':
    dataset = C10Dataset(batchsize=8)
    input_layer = Input((8, 3072))
    d1_layer = FullyConnected(input_layer, 4096)
    a1_layer = Relu(d1_layer)
    d2_layer = FullyConnected(a1_layer, 1024)
    a2_layer = Relu(d2_layer)
    d3_layer = FullyConnected(a2_layer, 128)
    a3_layer = CrossEntropyWithSoftmax(d3_layer)
    model = Model([input_layer, d1_layer, a1_layer,
                   d2_layer, a2_layer, d3_layer, a3_layer],
                  dataset)
    model.train()


