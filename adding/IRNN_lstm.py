
# coding: utf-8

# In[29]:
import cPickle as cp
import lasagne
from lasagne import layers
#from ../src.utils import load_mnist, init_constant, init_diagnal, shared_dataset
import numpy as np
import theano
import theano.tensor as T
import time
#import gru_layers

# In[43]:
# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10


print("Building network ...")
# First, we build the network, starting with an input layer
# Recurrent layers expect input of shape
# (batch size, max sequence length, number of features)
l_in = layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, 2))
# The network also needs a way to provide a mask for each sequence.  We'll
# use a separate input layer for that.  Since the mask only determines
# which indices are part of the sequence for each batch entry, they are
# supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
l_mask = layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
# We're using a bidirectional network, which means we will combine two
# RecurrentLayers, one with the backwards=True keyword argument.
# Setting a value for grad_clipping will clip the gradients in the layer
# Setting only_return_final=True makes the layers only return their output
# for the final time step, which is all we need for this task
l_forward = layers.LSTMLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    only_return_final=True)
l_backward = layers.LSTMLayer(
    l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
    only_return_final=True, backwards=True)
# Now, we'll concatenate the outputs to combine them.
l_concat = layers.ConcatLayer([l_forward, l_backward])
# Our output layer is a simple dense connection, with 1 output unit
l_out = layers.DenseLayer(
    l_concat, num_units=1, nonlinearity=lasagne.nonlinearities.tanh)

target_values = T.vector('target_output')

# lasagne.layers.get_output produces a variable for the output of the net
network_output = lasagne.layers.get_output(l_out)
# The network output will have shape (n_batch, 1); let's flatten to get a
# 1-dimensional vector of predicted values
predicted_values = network_output.flatten()
# Our cost will be mean-squared error
cost = T.mean((predicted_values - target_values)**2)
# Retrieve all parameters from the network
all_params = lasagne.layers.get_all_params(l_out)
sum = 0
for p in all_params:
    shape = p.shape.eval()
    print shape
    if len(shape) > 1:
        sum += shape[0] * shape[1]
    else:
        sum += shape[0]

print("sum params:%d"%sum)

# Compute SGD updates for training
print("Computing updates ...")
updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
# Theano functions for training and computing cost
print("Compiling functions ...")
train = theano.function([l_in.input_var, target_values, l_mask.input_var],
                        cost, updates=updates)
compute_cost = theano.function(
    [l_in.input_var, target_values, l_mask.input_var], cost)

# We'll use this "validation set" to periodically check progress
f = open('../data/adding.pkl')
test_data,train_data = cp.load(f)

X_val, y_val, mask_val = test_data

print("Training ...")
try:
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for i in range(EPOCH_SIZE):
            X, y, m = train_data[i]
            train(X, y, m)
        print("{:.3f}".format(time.time() - start_time))
        cost_val = compute_cost(X_val, y_val, mask_val)
        #print("Epoch {} validation cost = {}".format(epoch, cost_val))
except KeyboardInterrupt:
    pass
