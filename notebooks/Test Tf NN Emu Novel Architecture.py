
# coding: utf-8

# In[68]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[69]:


from pearce.emulator import OriginalRecipe, ExtraCrispy
from pearce.mocks import cat_dict
import numpy as np
from os import path


# In[70]:


import tensorflow as tf


# In[71]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()


# In[72]:


import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()


# In[73]:


training_file = '/home/sean/PearceRedMagicXiCosmoFixedNd.hdf5'
test_file = '/home/sean/PearceRedMagicXiCosmoFixedNd_test.hdf5'


# In[74]:


em_method = 'nn'
split_method = 'random'


# In[75]:


a = 1.0
z = 1.0/a - 1.0

emu.scale_bin_centers
# In[76]:


fixed_params = {'z':z}#, 'r': 24.06822623}

n_leaves, n_overlap = 50, 1
emu = ExtraCrispy(training_file, n_leaves, n_overlap, split_method, method = em_method, fixed_params=fixed_params,
                 custom_mean_function = None, downsample_factor = 1.0)
# In[77]:


emu = OriginalRecipe(training_file, method = em_method, fixed_params=fixed_params,
                    hyperparams = {'hidden_layer_sizes': (10),
                                 'activation': 'relu', 'verbose': True, 
                                    'tol': 1e-8, 'learning_rate_init':1e-4,\
                                   'max_iter':10, 'alpha':0, 'early_stopping':False, 'validation_fraction':0.3})


# In[78]:


x_train, y_train,yerr_train = emu.x ,emu.y ,emu.yerr


# In[79]:


_x_test, y_test, _, info = emu.get_data(test_file, fixed_params, None, False)
# whtien
x_test = (_x_test - _x_test.mean(axis = 0))/_x_test.std(axis = 0)


# In[80]:


y_train_mean, y_train_std = y_train.mean(axis = 0), y_train.std(axis =0)
#y_test_mean, y_test_std = y_test.mean(axis = 0), y_test.std(axis = 0)

y_train = (y_train- y_train_mean)/(y_train_std)
#y_test = (y_test - y_test_mean)/(y_test_std)


# In[81]:


x_train.shape, x_test.shape


# In[82]:


yerr_train[yerr_train==0] = 1


# In[83]:


def n_layer_fc(x, hidden_sizes, training=False, l = 1e-8, p=0.1):
    initializer = tf.variance_scaling_initializer(scale=2.0)
    regularizer = tf.contrib.layers.l2_regularizer(l)
    fc_output = tf.layers.dense(x, hidden_sizes[0], activation=tf.nn.relu,
                                 kernel_initializer = initializer, kernel_regularizer = regularizer)
                                 #kernel_regularizer = tf.nn.l2_loss)
    #fc2_output = tf.layers.dense(fc1_output, hidden_sizes[1], activation=tf.nn.relu,
    #                             kernel_initializer = initializer, kernel_regularizer = regularizer)
    for size in hidden_sizes[1:]:
        fc_output = tf.layers.dense(fc_output, size, kernel_initializer=initializer,
                                 kernel_regularizer = regularizer)
        bd_out = tf.layers.dropout(fc_output, p, training = training)
        bn_out = tf.layers.batch_normalization(bd_out, axis = -1, training=training)
        fc_output = tf.nn.relu(bn_out)#tf.nn.leaky_relu(bn_out, alpha=0.01)
        
    pred = tf.layers.dense(fc_output, 1, kernel_initializer=initializer, 
                              kernel_regularizer = regularizer)[:,0]#,
    return pred

def novel_fc(x, hidden_sizes, training=False, l = (1e-6, 1e-6, 1e-6), p = (0.5, 0.5, 0.5),\
             n_cosmo_params = 7, n_hod_params = 4):
    
    cosmo_sizes, hod_sizes, cap_sizes = hidden_sizes
    
    if type(l) is float:
        cosmo_l, hod_l, cap_l = l, l, l
    else:
        cosmo_l, hod_l, cap_l = l
        
    if type(p) is float:
        cosmo_p, hod_p, cap_p = p,p,p
    else:
        cosmo_p, hod_p, cap_p = p
    
    initializer = tf.variance_scaling_initializer(scale=2.0)
    
    #onlly for duplicating r
    n_params = n_cosmo_params+n_hod_params
    cosmo_x = tf.slice(x, [0,0], [-1, n_cosmo_params])
    cosmo_x = tf.concat(values=[cosmo_x, tf.slice(x, [0, n_params-1], [-1, -1]) ], axis = 1)
    #print tf.shape(cosmo_x)
    #print tf.shape(tf.slice(x, [0, n_params-1], [-1, -1]))
    hod_x = tf.slice(x, [0, n_cosmo_params], [-1, -1])
    
    cosmo_regularizer = tf.contrib.layers.l1_regularizer(cosmo_l)
    cosmo_out = cosmo_x
    
    for size in cosmo_sizes:
        fc_output = tf.layers.dense(cosmo_out, size,
                                 kernel_initializer = initializer,\
                                    kernel_regularizer = cosmo_regularizer)
        bd_out = tf.layers.dropout(fc_output, cosmo_p, training = training)
        bn_out = tf.layers.batch_normalization(bd_out, axis = -1, training=training)
        cosmo_out = tf.nn.relu(bn_out)#tf.nn.leaky_relu(bn_out, alpha=0.01)
    
    hod_regularizer = tf.contrib.layers.l1_regularizer(hod_l)
    hod_out = hod_x
    
    for size in hod_sizes:
        fc_output = tf.layers.dense(hod_out, size,
                                 kernel_initializer = initializer,\
                                    kernel_regularizer = hod_regularizer)
        bd_out = tf.layers.dropout(fc_output, hod_p, training = training)
        bn_out = tf.layers.batch_normalization(bd_out, axis = -1, training=training)
        hod_out = tf.nn.relu(bn_out)#tf.nn.leaky_relu(bn_out, alpha=0.01)
    
    cap_out=tf.concat(values=[cosmo_out, hod_out], axis = 1)
    cap_regularizer = tf.contrib.layers.l1_regularizer(cap_l)

    for size in cap_sizes:
        fc_output = tf.layers.dense(cap_out, size,
                                 kernel_initializer = initializer,\
                                    kernel_regularizer = cap_regularizer)
        bd_out = tf.layers.dropout(fc_output, cap_p, training = training)
        bn_out = tf.layers.batch_normalization(bd_out, axis = -1, training=training)
        cap_out = tf.nn.relu(bn_out)#tf.nn.leaky_relu(bn_out, alpha=0.01)
    
    pred = tf.layers.dense(cap_out, 1, kernel_initializer=initializer, 
                              kernel_regularizer = cap_regularizer)[:,0]#,
    return pred
# In[84]:


def optimizer_init_fn(learning_rate = 1e-6):
    return tf.train.AdamOptimizer(learning_rate)#, beta1=0.9, beta2=0.999, epsilon=1e-6)


# In[85]:


from sklearn.metrics import r2_score, mean_squared_error


# In[86]:


def check_accuracy(sess, val_data,batch_size, x, weights, preds, is_training=None):
    val_x, val_y = val_data
    perc_acc, scores = [],[]
    for idx in xrange(0, val_x.shape[0], batch_size):
        feed_dict = {x: val_x[idx:idx+batch_size],
                     is_training: 0}
        y_pred = sess.run(preds, feed_dict=feed_dict)*y_train_std + y_train_mean

        score = r2_score(val_y[idx:idx+batch_size], y_pred)
        scores.append(score)
        
        perc_acc = np.mean(np.abs(10**(val_y[idx:idx+batch_size])-10**(y_pred))/                           10**(val_y[idx:idx+batch_size]) )

        
    print 'Val score: %.6f, %.2f %% Loss'%(np.mean(np.array(scores)), 100*np.mean(np.array(perc_acc)))


# In[ ]:


device = '/device:GPU:0'
#device = '/cpu:0'
def train(model_init_fn, optimizer_init_fn,num_params, train_data, val_data, hidden_sizes,               num_epochs=1, batch_size = 200, l = 1e-6, p = 0.5, print_every=10):
    tf.reset_default_graph()    
    with tf.device(device):
        # Construct the computational graph we will use to train the model. We
        # use the model_init_fn to construct the model, declare placeholders for
        # the data and labels
        x = tf.placeholder(tf.float32, [None,num_params])
        y = tf.placeholder(tf.float32, [None])
        weights = tf.placeholder(tf.float32, [None])
        
        is_training = tf.placeholder(tf.bool, name='is_training')
        
        preds = model_init_fn(x, hidden_sizes, is_training, l = l)#, p=p)

        # Compute the loss like we did in Part II
        #loss = tf.reduce_mean(loss)
        
    with tf.device('/cpu:0'):
        loss = tf.losses.mean_squared_error(labels=y,                    predictions=preds, weights = weights)#weights?
        #loss = tf.losses.absolute_difference(labels=y, predictions=preds, weights = tf.abs(1.0/y))#weights?

    with tf.device(device):
        optimizer = optimizer_init_fn()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        
    with tf.Session() as sess:
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
                check_accuracy(sess, val_data, batch_size, x, weights, preds, is_training=is_training)
            #t += 1


# In[ ]:


train(n_layer_fc, optimizer_init_fn, x_train.shape[1],           (x_train, y_train, yerr_train), (x_test, y_test),           [100, 200, 500, 200, 100], num_epochs= int(1e4),           batch_size = 2000, l = 1e-7, p = 0.0,           print_every = 200)


# In[ ]:


np.abs(emu.goodness_of_fit(training_file, statistic = 'log_frac')).mean()


# In[ ]:


np.abs(emu.goodness_of_fit(training_file, statistic = 'frac')).mean()


# In[ ]:


fit_idxs = np.argsort(gof.mean(axis = 1))


# In[ ]:


emu.goodness_of_fit(training_file).mean()#, statistic = 'log_frac')).mean()


# In[ ]:


model = emu._emulator


# In[ ]:


ypred = model.predict(emu.x)


# In[ ]:


plt.hist( np.log10( (emu._y_std+1e-5)*np.abs(ypred-emu.y)/np.abs((emu._y_std+1e-5)*emu.y+emu._y_mean) ))


# In[ ]:


( (emu._y_std+1e-5)*np.abs(ypred-emu.y)/np.abs((emu._y_std+1e-5)*emu.y+emu._y_mean) ).mean()


# In[ ]:


emu._y_mean, emu._y_std


# In[ ]:


for idx in fit_idxs[:10]:
    print gof[idx].mean()
    print (ypred[idx*emu.n_bins:(idx+1)*emu.n_bins]-emu.y[idx*emu.n_bins:(idx+1)*emu.n_bins])/emu.y[idx*emu.n_bins:(idx+1)*emu.n_bins]
    plt.plot(emu.scale_bin_centers, ypred[idx*emu.n_bins:(idx+1)*emu.n_bins], label = 'Emu')
    plt.plot(emu.scale_bin_centers, emu.y[idx*emu.n_bins:(idx+1)*emu.n_bins], label = 'True')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.show()


# In[ ]:


print dict(zip(emu.get_param_names(), emu.x[8*emu.n_bins, :]*emu._x_std+emu._x_mean))


# In[ ]:


emu.get_param_names()

#print emu.x.shape
#print emu.downsample_x.shape
if hasattr(emu, "_emulators"):
    print emu._emulators[0]._x.shape
else:
    print emu._emulator._x.shape
# In[ ]:


emu._ordered_params

x, y, y_pred = emu.goodness_of_fit(training_file, statistic = 'log_frac')x, y, y_predN = 25
for _y, yp in zip(y[:N], y_pred[:N]):
    #plt.plot(emu.scale_bin_centers ,  (_y - yp)/yp ,alpha = 0.3, color = 'b')
    
    plt.plot(emu.scale_bin_centers, _y, alpha = 0.3, color = 'b')
    plt.plot(emu.scale_bin_centers, yp, alpha = 0.3, color = 'r')
    
plt.loglog();%%timeit
#truth_file = '/u/ki/swmclau2/des/PearceRedMagicWpCosmoTest.hdf5'
gof = emu.goodness_of_fit(training_file, N = 100, statistic = 'log_frac')
# In[ ]:


gof = emu.goodness_of_fit(training_file, statistic = 'frac')
print gof.mean()


# In[ ]:


for row in gof:
    print row


# In[ ]:


gof = emu.goodness_of_fit(training_file, statistic = 'frac')
print gof.mean()


# In[ ]:


model = emu._emulator


# In[ ]:


model.score(emu.x, emu.y)


# In[ ]:


ypred = model.predict(emu.x)

np.mean(np.abs(ypred-emu.y)/emu.y)


# In[ ]:


plt.plot(emu.scale_bin_centers, np.abs(gof.mean(axis = 0)) )
plt.plot(emu.scale_bin_centers, np.ones_like(emu.scale_bin_centers)*0.01)
plt.plot(emu.scale_bin_centers, np.ones_like(emu.scale_bin_centers)*0.05)
plt.plot(emu.scale_bin_centers, np.ones_like(emu.scale_bin_centers)*0.1)


plt.loglog();


# In[ ]:


plt.plot(emu.scale_bin_centers, np.abs(gof.T),alpha = 0.1, color = 'b')
plt.plot(emu.scale_bin_centers, np.ones_like(emu.scale_bin_centers)*0.01, lw = 2, color = 'k')
plt.loglog();


# In[ ]:


gof[:,i].shape


# In[ ]:


for i in xrange(gof.shape[1]):
    plt.hist(np.log10(gof[:, i]), label = str(i), alpha = 0.2);
    
plt.legend(loc='best')
plt.show()

