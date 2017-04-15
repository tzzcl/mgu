import theano
import theano.tensor as T
import numpy as np
from layers import FullConnectLayer

class AutoEncode:
    def __init__(self, x, n_in, n_hidden, n_layer):
        self.x = x
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_layer = n_layer
        self.params = []

        layer1 = FullConnectLayer(self.x, n_in, n_hidden)
        self.params += layer1.params

        last_layer = layer1
        for i in range(1, n_layer-1):
            layer = FullConnectLayer(last_layer.output, n_hidden, n_hidden)
            self.params += layer.params
            last_layer = layer

        layer_final = FullConnectLayer(last_layer.output, n_hidden, n_in)
        self.params += layer_final.params
        self.x_hat = layer_final.output

    def get_cost_updates(self, learning_rate):
        L = - T.sum(self.x * T.log(self.x_hat) + (1 - self.x) * T.log(1 - self.x_hat), axis=1)
        cost = T.mean(L)

        gparams = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return cost, updates

    def get_reconstructed_input(self):
        return self.x_hat
