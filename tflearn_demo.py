# -*- coding: utf-8 -*-
"""
Example on how to use HDF5 dataset with TFLearn. HDF5 is a data model,
library, and file format for storing and managing data. It can handle large
dataset that could not fit totally in ram memory. Note that this example
just give a quick compatibility demonstration. In practice, there is no so
real need to use HDF5 for small dataset such as CIFAR-10.
"""

from __future__ import division, print_function, absolute_import

import h5py
from tflearn.data_utils import *
from tflearn.layers.conv import *
from tflearn.layers.core import *
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import *

base_dir = os.path.expanduser("/home/utopia/CVDATA/German_AI_Challenge_2018/session1")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_s18_train = os.path.join(base_dir, 's18_train.h5')
path_s18_val = os.path.join(base_dir, 's18_val.h5')

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
s18_train = h5py.File(path_s18_train, 'r')
s18_val = h5py.File(path_s18_val, 'r')

X = s18_train['s18_train']
X_test = s18_val['s18_val']
Y = fid_training['label']
Y_test = fid_validation['label']
print(X.shape, Y.shape)
print(X_test.shape, Y_test.shape)

# Build network
network = input_data(shape=[None, 32, 32, 18], dtype=tf.float32)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.1)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
# model.load("lcz42_20181218_epoch10.tfl")
model.fit(X, Y, n_epoch=10, shuffle=True,
          validation_set=(X_test, Y_test), show_metric=True, batch_size=512,
          run_id='lcz42')

model.save("lcz42_20181219_epoch10.tfl")

fid_training.close()
fid_validation.close()
s18_train.close()
s18_val.close()
