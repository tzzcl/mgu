#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle as cp


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


def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH, n_batch=N_BATCH):
    '''
    Generate a batch of sequences for the "add" task, e.g. the target for the
    following

    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``

    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.

    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    n_batch : int
        Number of samples in the batch.

    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (n_batch, max_length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample, shape (n_batch,).
    mask : np.ndarray
        A binary matrix of shape (n_batch, max_length) where ``mask[i, j] = 1``
        when ``j <= (length of sequence i)`` and ``mask[i, j] = 0`` when ``j >
        (length of sequence i)``.

    References
    ----------
    .. [1] Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory."
    Neural computation 9.8 (1997): 1735-1780.

    .. [2] Sutskever, Ilya, et al. "On the importance of initialization and
    momentum in deep learning." Proceedings of the 30th international
    conference on machine learning (ICML-13). 2013.
    '''
    # Generate X - we'll fill the last dimension later
    X = np.concatenate([np.random.uniform(size=(n_batch, max_length, 1)),
                        np.zeros((n_batch, max_length, 1))],
                       axis=-1)
    mask = np.zeros((n_batch, max_length))
    y = np.zeros((n_batch,))
    # Compute masks and correct values
    for n in range(n_batch):
        # Randomly choose the sequence length
        length = np.random.randint(min_length, max_length)
        # Make the mask for this sample 1 within the range of length
        mask[n, :length] = 1
        # Zero out X after the end of the sequence
        X[n, length:, 0] = 0
        # Set the second dimension to 1 at the indices to add
        X[n, np.random.randint(length/10), 1] = 1
        X[n, np.random.randint(length/2, length), 1] = 1
        # Multiply and sum the dimensions of X to get the target value
        y[n] = np.sum(X[n, :, 0]*X[n, :, 1])
    # Center the inputs and outputs
    X -= X.reshape(-1, 2).mean(axis=0)
    y -= y.mean()
    return (X.astype(theano.config.floatX), y.astype(theano.config.floatX),
            mask.astype(theano.config.floatX))


if __name__ == '__main__':
    test = gen_data()
    train = []
    for _ in range(EPOCH_SIZE):
        train.append(gen_data())


    f = open('adding.pkl','wb')
    cp.dump((test,train),f,2)
    f.close()

