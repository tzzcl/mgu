import numpy as np
import theano
import gzip
import cPickle
import theano.tensor as T
from random import shuffle
from random import randint


def load_mnist(mnist_path):
    f = gzip.open(mnist_path)
    train_set, valid_set, test_set = cPickle.load(f)

    return train_set, valid_set, test_set

def expandData(X,y):
    newX = []
    labels = []
    null = [1]*230
    for line,label in zip(X,y):
        for i in xrange(30):
            shuffle(line)
            labels.append(label)
            newline = []
            for i in range(20):
                if i < len(line):
                    newline.extend(line[i])
                else:
                    newline.extend(line[randint(0,len(line) - 1)])
            newX.append(newline)
    return (newX,labels)


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def init_diagnal(size, name=None, scale=1):
    init_w = np.asarray(
        np.identity(size) * scale,
        dtype=theano.config.floatX
    )
    return theano.shared(value=init_w, name=name, borrow=True)


def init_uniform(size, name=None, low=-0.01, high=0.01):
    init_w = np.asarray(
        np.random.uniform(
            size=size,
            low=low,
            high=high,
        ),
        dtype=theano.config.floatX
    )
    return theano.shared(value=init_w, name=name, borrow=True)


def init_constant(size, name=None, value=0):
    init_v = np.asarray(
        np.ones(size) * value,
        dtype=theano.config.floatX
    )
    return theano.shared(value=init_v, name=name, borrow=True)
