import h5py
import keras
import os
from keras.applications.nasnet import NASNetMobile
from keras.callbacks import TensorBoard

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

# x_train = s18_train['s18_train']
# x_test = s18_val['s18_val']
x_train = s2_train_index['s2_train_index']
x_test = s2_val_index['s2_val_index']
y_train = fid_training['label']
y_test = fid_validation['label']

model = NASNetMobile(input_shape=(32, 32, 3),
                     # include_top=True,
                     weights=None,  # 'imagenet'
                     # input_tensor=None,
                     # pooling=None,
                     classes=17)

# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
# opt = keras.optimizers.SGD(lr=1e-1, momentum=0.0, decay=0.0, nesterov=False)
# opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
log_path = '/tmp/tflearn_logs/NASNetMobile_LCZ42_Adadelta'
callback = TensorBoard(log_path)
callback.set_model(model)

model.fit(x_train, y_train,
          batch_size=1024,  # 128,1024
          epochs=10,
          shuffle="batch",
          validation_data=(x_test, y_test)
          )

modelpath = 'NASNetMobile_Adadelta_epochs_10.h5'
model.save(modelpath)
print('Saved trained model at %s ' % modelpath)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
