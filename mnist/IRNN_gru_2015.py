
# coding: utf-8

# In[29]:
import lasagne
from lasagne import layers
from src.utils import load_mnist, init_constant, init_diagnal, shared_dataset
#load_mnist, init_constant, init_diagnal, shared_dataset
import numpy as np
import theano
import theano.tensor as T
import time
from MGULayer import MGULayer


# In[39]:

train_set, valid_set, test_set = load_mnist('../data/mnist.pkl.gz')
print valid_set[0].shape
print test_set[0].shape

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

# In[43]:
seq_len = 784
feature_num = 1
hidden_unit = 100

train_x = train_x.reshape((-1, seq_len, feature_num))
valid_x = valid_x.reshape((-1, seq_len, feature_num))
test_x = test_x.reshape((-1, seq_len, feature_num))

train_x, train_y = shared_dataset((train_x, train_y))
valid_x, valid_y = shared_dataset((valid_x, valid_y))
test_x, test_y = shared_dataset((test_x, test_y))


# In[46]:

input_var = T.tensor3('inputs')
target_var = T.ivector('targets')
index = T.iscalar("index")
batch_size = 500
n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size
n_test_batches = test_x.get_value(borrow=True).shape[0] / batch_size

l_in = layers.InputLayer(shape=(None, seq_len, feature_num), input_var=input_var)
#l_rec = layers.RecurrentLayer(incoming=l_in,
#                              num_units=100,
#                              W_hid_to_hid=init_diagnal(100),
#                              b=init_constant(size=(100,)),
#                              nonlinearity=lasagne.nonlinearities.rectify,
#                              grad_clipping=1)
l_rec = MGULayer(incoming=l_in,
                              num_units=hidden_unit)
l_out = layers.DenseLayer(incoming=l_rec,
                          num_units=10,
                          nonlinearity=lasagne.nonlinearities.softmax)

prediction = lasagne.layers.get_output(l_out)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

params = lasagne.layers.get_all_params(l_out, trainable=True)
sum = 0
for p in params:
    shape = p.shape.eval()
    print shape
    if len(shape) > 1:
        sum += shape[0] * shape[1]
    else:
        sum += shape[0]

print("sum params:%d"%sum)
updates = lasagne.updates.nesterov_momentum(
    loss, params, learning_rate=1e-8, momentum=0.99)

test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)

train_fn = theano.function([index],
                           loss,
                           updates=updates,
                           givens={
    input_var: train_x[index * batch_size: (index + 1) * batch_size],
    target_var: train_y[index * batch_size: (index + 1) * batch_size]
})

val_fn = theano.function([index],
                         [test_loss, test_acc],
                         givens={
    input_var: valid_x[index * batch_size: (index + 1) * batch_size],
    target_var: valid_y[index * batch_size: (index + 1) * batch_size]
})

num_epochs = 1000000
valid_feq = 10
print "start training...."
f = open('./result/result_mru.csv','w',0)

for epoch in range(num_epochs):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in range(n_train_batches):
        train_err += train_fn(batch)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    if epoch % valid_feq == 0 and epoch > 0:
        for batch in range(n_valid_batches):
            err, acc = val_fn(batch)
            val_err += err
            val_acc += acc
            val_batches += 1

    # Then we print the results for this epoch:
    f.write("Epoch {} of {} took {:.3f}s\n".format(
        epoch + 1, num_epochs, time.time() - start_time))
    f.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
    if epoch % valid_feq == 0 and epoch > 0:
        f.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
        f.write("  validation accuracy:\t\t{:.2f} %\n".format(
            val_acc / val_batches * 100))


# In[ ]:
