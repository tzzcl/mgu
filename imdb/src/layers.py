import theano
import theano.tensor as T
import numpy as np
from utils import *


class FullConnectLayer:
    def __init__(self, input, n_in, n_out, activation=T.nnet.sigmoid):
        self.input = input
        self.W = init_uniform(
            size=(n_in, n_out),
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out))
        )

        self.b = init_constant(size=n_out)

        self.output = activation(T.dot(self.input, self.W) + self.b)

        self.params = [self.W, self.b]

class Regression:
    def __init__(self, input, n_in, n_out):
        self.input = input
        self.output = self.input
        self.params = []
        # self.W = init_uniform(
        #     size=(n_in, n_out),
        #     low=-np.sqrt(6. / (n_in + n_out)),
        #     high=np.sqrt(6. / (n_in + n_out))
        # )
        #
        # self.b = init_constant(size=n_out)

        # self.output = T.dot(self.input, self.W) + self.b

        #self.params = [self.W, self.b]


if __name__ == '__main__':
    x = T.matrix()
    f = FullConnectLayer(x, 225*225*3, 20)
    print f.W.get_value()
    print f.b.get_value()


