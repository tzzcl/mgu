import theano
import theano.tensor as T
import numpy as np
from layers import FullConnectLayer, Regression

def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class ImagePaint:
    def __init__(self, x, n_in, n_hidden, n_layer):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.params = []

        layer1 = FullConnectLayer(self.x, n_in, n_hidden, activation=relu)
        self.params += layer1.params

        last_layer = layer1
        for i in range(1, n_layer-1):
            layer = FullConnectLayer(last_layer.output, n_hidden, n_hidden)
            self.params += layer.params
            last_layer = layer

        layer = FullConnectLayer(last_layer.output, n_hidden, 3)
        self.params += layer.params
        last_layer = layer

        layer_final = Regression(last_layer.output, 3, 3)
        self.params += layer_final.params
        self.y_hat = layer_final.output

    def get_cost_updates(self, y, learning_rate):
        L = T.sum(T.pow(y-self.y_hat, 2), axis=1)
        cost = T.mean(L) / 2

        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return cost, updates

    def get_reconstructed_input(self):
        return self.y_hat
