from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path
import tensorflow as tf
import keras
from tensorflow.python.client import device_lib
import sys
from time import time

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
print get_available_gpus()

#xi gg
training_file = 'PearceXiggChinchilla.hdf5'
#training_file = 'PearceRedMagicXiCosmoFixedNd.hdf5'
#test_file= '/scratch/users/swmclau2/xi_zheng07_cosmo_test_lowmsat2/'
test_file =  'PearceXiggChinchillaTest.hdf5'

#em_method = 'nn'
#split_method = 'random'

#a = 1.0
#z = 1.0/a - 1.0
#array([  0.09581734,   0.13534558,   0.19118072,   0.27004994,
#         0.38145568,   0.53882047,   0.76110414,   1.07508818,
#                  1.51860241,   2.14508292,   3.03001016,   4.28000311,
#                           6.04566509,   8.53972892,  12.06268772,  17.0389993 ,
#                                   24.06822623,  33.99727318])
#fixed_params = {'z':z}#, 'r': 0.09581734}

# This is just me loading the data with my own code, nothing to do with NN

# TODO ostrich module that just gets the data in the write format.
print 'Loading training data'
#emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,
#                    hyperparams = {'hidden_layer_sizes': (2),
#                                 'activation': 'relu', 'verbose': True, 
#                                    'tol': 1e-8, 'learning_rate_init':1e-4,\
#                                   'max_iter':2, 'alpha':0, 'early_stopping':False, 'validation_fraction':0.9})

#x_train, y_train,yerr_train = emu.x, emu.y, emu.yerr
#x_test, y_test,yerr_test = emu.x, emu.y, emu.yerr

print 'Loading test data'
#_x_test, y_test, _, info = emu.get_data(test_file, fixed_params, None, False)
# whiten the data
#x_test = (_x_test - _x_test.mean(axis = 0))/_x_test.std(axis = 0)

#np.save('x_train.npy', x_train)
#np.save('y_train.npy', y_train)
#np.save('yerr_train.npy', yerr_train)

#np.save('x_test.npy', x_test)
#np.save('y_test.npy', y_test)
x_train = np.load('x_train.npy')#[:N]
y_train = np.load('y_train.npy')#[:N]
yerr_train = np.load('yerr_train.npy')#[:N]

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

#yerr_train[yerr_train==0] = 1
y_train_mean, y_train_std = 0.0, 1.0#y_train.mean(axis = 0), 1.0 
y_test_mean, y_test_std = 0.0, 1.0#y_test.mean(axis = 0), 1.0 

y_train = (y_train- y_train_mean)/(y_train_std)
# don't do to y_data
y_test = (y_test - y_test_mean)/(y_test_std)

# function that defines a Fully connected network with N layers
def n_layer_fc(x, hidden_sizes, training=False, l = 1e-8):#, p=0.1):
    initializer = tf.variance_scaling_initializer(scale=2.0)
    #regularizer = tf.contrib.layers.l2_regularizer(l)
    fc_output = tf.layers.dense(x, hidden_sizes[0], activation=tf.nn.relu,
                                 kernel_initializer = initializer)#, kernel_regularizer = regularizer)
                                 #kernel_regularizer = tf.nn.l2_loss)
    #fc2_output = tf.layers.dense(fc1_output, hidden_sizes[1], activation=tf.nn.relu,
    #                             kernel_initializer = initializer, kernel_regularizer = regularizer)
    for size in hidden_sizes[1:]:
        fc_output = tf.layers.dense(fc_output, size,activation = tf.nn.relu,  kernel_initializer=initializer)#,
                                 #kernel_regularizer = regularizer)
        #bd_out = tf.layers.dropout(fc_output, p, training = training)
        #bn_out = tf.layers.batch_normalization(bd_out, axis = -1, training=training)
        #fc_output = tf.nn.relu(bn_out)#tf.nn.leaky_relu(bn_out, alpha=0.01)
        
    pred = tf.layers.dense(fc_output, 1, kernel_initializer=initializer)[:,0]#, 
                              #kernel_regularizer = regularizer)[:,0]#,
    return pred

# In[84]:


# Optimizier function
def optimizer_init_fn(learning_rate = 1e-4):
    return tf.train.AdamOptimizer(learning_rate)#, beta1=0.9, beta2=0.999, epsilon=1e-6)


from sklearn.metrics import r2_score, mean_squared_error


# check the loss and % accuracy of the predictions
def check_accuracy(sess, val_data,batch_size, x, weights, preds, is_training=None):
    val_x, val_y = val_data
    perc_acc, scores = [],[]
    for idx in xrange(0, val_x.shape[0], batch_size):
        feed_dict = {x: val_x[idx:idx+batch_size],
                     is_training: 0}
        y_pred = sess.run(preds, feed_dict=feed_dict)*y_train_std + y_train_mean

        score = r2_score(val_y[idx:idx+batch_size], y_pred)
        scores.append(score)
        y_true = val_y[idx:idx+batch_size]*y_train_std+y_train_mean
        perc_acc = np.mean(np.abs(10**(y_true)-10**(y_pred))/                           10**(y_true) )

        
    print 'Val score: %.6f, %.2f %% Loss'%(np.mean(np.array(scores)), 100*np.mean(np.array(perc_acc)))

#device = '/cpu:0'
# main training function
def train(model_init_fn, optimizer_init_fn,num_params, train_data, val_data, hidden_sizes,               num_epochs=1, batch_size = 200, l = 1e-6, print_every=10):
    tf.reset_default_graph()    
    # Construct the computational graph we will use to train the model. We
    # use the model_init_fn to construct the model, declare placeholders for
    # the data and labels
    x = tf.placeholder(tf.float32, [None,num_params])
    y = tf.placeholder(tf.float32, [None])
    weights = tf.placeholder(tf.float32, [None])
    
    is_training = tf.placeholder(tf.bool, name='is_training')
    with tf.device('/device:GPU:0'): 
        preds = model_init_fn(x, hidden_sizes, is_training, l = l)#, p=p)

        # Compute the loss like we did in Part II
        #loss = tf.reduce_mean(loss)
            
        loss = tf.losses.mean_squared_error(labels=y, predictions=preds, weights = weights)#weights?
        #loss = tf.losses.absolute_difference(labels=y, predictions=preds, weights = weights)#weights?

        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
            
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, \
                                          log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        #t = 0
        train_x, train_y, train_yerr = train_data
        rand_idxs = range(train_x.shape[0])
        for epoch in range(num_epochs):
            #print('Starting epoch %d' % epoch)
            np.random.shuffle(rand_idxs)
            losses = []
            for idx in xrange(0, train_x.shape[0], batch_size):
                feed_dict = {x: train_x[rand_idxs[idx:idx+batch_size]],                             y: train_y[rand_idxs[idx:idx+batch_size]],                             weights: 1/train_yerr[rand_idxs[idx:idx+batch_size]],                             is_training:1}
                loss_np, _, preds_np = sess.run([loss, train_op, preds], feed_dict=feed_dict)
                losses.append(loss_np)
            if epoch % print_every == 0:
                loss_avg = np.mean(np.array(losses))
                print('Epoch %d, loss = %e' % (epoch, loss_avg))
                sys.stdout.flush()
                check_accuracy(sess, val_data, batch_size, x, weights, preds, is_training=is_training)
            #t += 1


#print fixed_params
#sizes = [100, 250, 500, 250, 100]#, 2000, 1000]#, 100]
sizes = [100, 100]#, 100]#,10]
bs = 100 
#l, p = 1e-6, 0.1
l = 0.0
print 'Sizes', sizes
print 'Batch size', bs
print 'Regularization:', l

for i in [1,2,5,10,20,50]:
    t0 = time()
    N = int(i*1e3*18)
    print 'Npoints: ', N
    indices = np.random.choice(x_train.shape[0], N, replace = False)
    train(n_layer_fc, optimizer_init_fn, x_train.shape[1], (x_train[indices], y_train[indices], yerr_train[indices]), (x_test, y_test), sizes, num_epochs= 200,           batch_size = bs, l = l, print_every = 10)
    print 'Total Time: ', time()-t0, ' seconds'
    print '*'*30

