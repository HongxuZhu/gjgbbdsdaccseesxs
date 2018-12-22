# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import tflearn
import h5py, os

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Data loading
# from tflearn.datasets import cifar10
# (X, Y), (testX, testY) = cifar10.load_data()
# Y = tflearn.data_utils.to_categorical(Y)
# testY = tflearn.data_utils.to_categorical(testY)

base_dir = os.path.expanduser("/home/utopia/CVDATA/German_AI_Challenge_2018/session1")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_s18_train = os.path.join(base_dir, 's18_train.h5')
path_s18_val = os.path.join(base_dir, 's18_val.h5')
path_s2_train_index = os.path.join(base_dir, 's2_train_index.h5')
path_s2_val_index = os.path.join(base_dir, 's2_val_index.h5')

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')
s18_train = h5py.File(path_s18_train, 'r')
s18_val = h5py.File(path_s18_val, 'r')
s2_train_index = h5py.File(path_s2_train_index, 'r')
s2_val_index = h5py.File(path_s2_val_index, 'r')

# X = s18_train['s18_train']
# X_test = s18_val['s18_val']

X = s2_train_index['s2_train_index']
X_test = s2_val_index['s2_val_index']
Y = fid_training['label']
Y_test = fid_validation['label']

# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([32, 32], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3]  # ,
                         # data_preprocessing=img_prep,
                         # data_augmentation=img_aug
                         )
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n - 1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n - 1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 17, activation='softmax')
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=opt, loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, tensorboard_verbose=0,
                    checkpoint_path='model_ResNeXt5_s2index_epoch20', max_checkpoints=100,
                    clip_gradients=0.)

# model.load('lcz42_20181218_ResNeXt5_epoch10.tfl')
print('Load pre-existing model, restoring all weights')
model.fit(X, Y, n_epoch=20, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=128, shuffle=True,
          run_id='ResNeXt5_s2index_epoch20')
model.save("lcz42_20181218_ResNeXt5_s2index_epoch20.tfl")

fid_training.close()
fid_validation.close()
s18_train.close()
s18_val.close()
