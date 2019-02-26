import numpy as np
from os.path import join 
from glob import glob
import tensorflow as tf

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print get_available_gpus()


data_dir = '/home/users/swmclau2/scratch/wp_data_trainingtest/'
cosmo_data_fname = 'cosmology_camb_full.dat'
hod_data_fname = 'HOD_design_np11_n5000.dat'

cosmo_colnames = ['Om', 'Ob', 'sigma_8', 'h', 'n_s']
cosmo_data = np.loadtxt(join(data_dir, cosmo_data_fname))#, delimiter='  ')

hod_colnames = ['M1', 'alpha', 'Mmin', 'sigma_logM']
hod_data = np.loadtxt(join(data_dir, hod_data_fname))#, delimiter = ' ')

mass_points = hod_data > 1e3 #I think this cut will work

hod_data[mass_points] = np.log10(hod_data[mass_points])

clustering_dir = 'wp_training_output'
clustering_files = sorted(glob(join(data_dir, clustering_dir) + '/*.dat') )

print len(clustering_files)

# TODO i want to do this for individual bins.
nbins = 9
bin_idx = 7 # second to last bin
zx = np.zeros((len(clustering_files), cosmo_data.shape[1]+hod_data.shape[1]))
zy = np.zeros((len(clustering_files),))

for i, cf in enumerate(clustering_files):
    if i%1000==0:
        print i
    if 'err' in cf:
        continue
    data = np.loadtxt(cf)#, delimiter = ' ')
    #rs = np.log10(data[:,0])
    wp = np.log10(data[:,1])
    fbase = cf.split('/')[-1]
    split_fbase = fbase.split('_')
    cosmo, hod = int(split_fbase[3]), int(split_fbase[5])
        
    n_cosmo_params = cosmo_data.shape[1]

    zx[i, :n_cosmo_params] = cosmo_data[cosmo]
    zx[i, n_cosmo_params:] = hod_data[hod]
    zy[i] = wp[bin_idx]


zx_mean, zx_std = zx.mean(axis =0), zx.std(axis = 0)
zx = (zx-zx_mean)/(zx_std+1e-5)

np.savetxt(join(data_dir,'zx.npy'), zx)
np.savetxt(join(data_dir,'zy.npy'), zy)

# repeat with test sets
cosmo_data_fname = 'cosmology_camb_test_box_full.dat'
hod_data_fname = 'HOD_test_np11_n1000.dat'

cosmo_data_test = np.loadtxt(join(data_dir, cosmo_data_fname))#, delimiter='  ')

hod_data_test = np.loadtxt(join(data_dir, hod_data_fname))#, delimiter = ' ')

mass_points = hod_data_test > 1e3 #I think this cut will work
hod_data_test[mass_points] = np.log10(hod_data_test[mass_points])

clustering_dir = 'wp_test_output_mean'
clustering_files = sorted(glob(join(data_dir, clustering_dir) + '/*.dat') )

zx_test = np.zeros((len(clustering_files)*nbins, cosmo_data.shape[1]+hod_data.shape[1]))
zy_test = np.zeros((len(clustering_files)*nbins,))

y_train_mean, y_train_std = 0.0, 1.0

for i, cf in enumerate(clustering_files):
    if i%1000==0:
        print i
    if 'err' in cf:
        continue
    data = np.loadtxt(cf, delimiter = ' ')
    #rs = np.log10(data[:,0])
    wp = np.log10(data[:,1])
    fbase = cf.split('/')[-1]
    split_fbase = fbase.split('_')
    cosmo, hod = int(split_fbase[3]), int(split_fbase[5])

    n_cosmo_params = cosmo_data.shape[1]
        
    zx_test[i, :n_cosmo_params] = cosmo_data_test[cosmo]
    zx_test[i, n_cosmo_params:] = hod_data_test[hod]
    zy_test[i] = wp[bin_idx]

zx_test = (zx_test-zx_mean)/(zx_std+1e-5)

np.savetxt(join(data_dir,'zx_test.npy'), zx_test)
np.savetxt(join(data_dir,'zy_test.npy'), zy_test)

print 'made the data'
# function that defines a Fully connected network with N layers
def n_layer_fc(x, hidden_sizes, training=False, l = 1e-5, p=0.1):
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
        fc_output = tf.nn.leaky_relu(bn_out)#, alpha=0.01)
        
    pred = tf.layers.dense(fc_output, 1, kernel_initializer=initializer, 
                              kernel_regularizer = regularizer)[:,0]#,
    return pred

# network similar to the above but with an architecture that splits cosmo and hod inputs
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


# Optimizier function
def optimizer_init_fn(learning_rate = 2e-7):
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
        
        perc_acc = np.mean(np.abs(10**(val_y[idx:idx+batch_size])-10**(y_pred))/                           10**(val_y[idx:idx+batch_size]) )

        
    print 'Val score: %.6f, %.2f %% Loss'%(np.mean(np.array(scores)), 100*np.mean(np.array(perc_acc)))

device = '/device:GPU:0'
#device = '/cpu:0'
# main training function
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


#sizes = [100, 250, 500, 250, 100]#, 2000, 1000]#, 100]
sizes = [500,500, 500]#,10]
bs = 100
l, p = 1e-5, 0.0

train(n_layer_fc, optimizer_init_fn, zx.shape[1], (zx, zy, np.ones_like(zy)), (zx_test, zy_test), sizes, num_epochs= int(1e5),           batch_size = bs, l = l, p = p, print_every = 2000)

