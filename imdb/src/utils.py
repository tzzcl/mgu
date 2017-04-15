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

def load_imdb(path="imdb.pkl", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here IMDB)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    train_set = cPickle.load(f)
    test_set = cPickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            idx = maxlen
            if len(x) < maxlen:
                idx = len(x)
            x_new = np.zeros(maxlen).astype('int64')
            x_new[:idx] = x[:idx]

            new_train_set_x.append(x_new)
            new_train_set_y.append(y)

        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

        new_test_set_x = []
        new_test_set_y = []
        for x, y in zip(test_set[0], test_set[1]):
            idx = maxlen
            if len(x) < maxlen:
                idx = len(x)
            x_new = np.zeros(maxlen).astype('int64')
            x_new[:idx] = x[:idx]

            new_test_set_x.append(x_new)
            new_test_set_y.append(y)

        test_set = (new_test_set_x, new_test_set_y)
        del new_test_set_x, new_test_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    test_set_x = np.asarray(test_set_x)
    train_set_x = np.asarray(train_set_x)
    valid_set_x = np.asarray(valid_set_x)

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)

    return train, valid, test

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
