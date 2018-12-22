import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report

### to change according to your machine
base_dir = os.path.expanduser("/home/utopia/CVDATA/German_AI_Challenge_2018/session1")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')

## we can have a look at which keys are stored in the file
## you will get the return [u'label', u'sen1', u'sen2']
## sen1 and sen2 means the satellite images
print(fid_training.keys())
print(fid_validation.keys())

### get s1 image channel data
### it is not really loaded into memory. only the indexes have been loaded.
print("-" * 60)
print("training part")
s1_training = fid_training['sen1']
print(s1_training.shape)
s2_training = fid_training['sen2']
print(s2_training.shape)
label_training = fid_training['label']
print(label_training.shape)

print("-" * 60)
print("validation part")
s1_validation = fid_validation['sen1']
print(s1_validation.shape)
s2_validation = fid_validation['sen2']
print(s2_validation.shape)
label_validation = fid_validation['label']
print(label_validation.shape)

label_qty = np.sum(label_training, axis=0)
plt.plot(label_qty)

plt.subplot(121)
plt.imshow(10 * np.log10(s1_training[0, :, :, 4]), cmap=plt.cm.get_cmap('gray'));
plt.colorbar()
plt.title('Sentinel-1')

plt.subplot(122)
plt.imshow(s2_training[0, :, :, 1], cmap=plt.cm.get_cmap('gray'));
plt.colorbar()
plt.title('Sentinel-2')

plt.show()

train_s1 = s1_training
train_s2 = s2_training
train_label = label_training

train_y = np.argmax(train_label, axis=1)
classes = list(set(train_y))
batch_size = 10000
# n_sampels = batch_size
n_sampels = int(train_s1.shape[0])

params = {
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 17,  # 类别数,与 multisoftmax 并用
    'tree_method': 'gpu_hist',
    'predictor': 'cpu_predictor',  # cpu_predictor,gpu_predictor
    'nthread': 4,
    'seed': 1987,
    # 'eval_metric': ['merror', 'mlogloss'],
    'verbosity': 0  # 打印消息的详细程度
}


def val(clf):
    print("-" * 60)
    pred_y = []
    train_val_y = np.argmax(label_validation, axis=1)
    n_val_samples = s2_validation.shape[0]
    for i in range(0, n_val_samples, batch_size):
        start_pos = i
        end_pos = min(i + batch_size, n_val_samples)
        val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
        val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
        cur_batch_size = val_s2_batch.shape[0]
        val_s1_batch = val_s1_batch.reshape((cur_batch_size, -1))
        val_s2_batch = val_s2_batch.reshape((cur_batch_size, -1))
        val_X_batch = np.hstack([val_s1_batch, val_s2_batch])
        tmp_pred_y = clf.predict(xgb.DMatrix(val_X_batch))
        pred_y.append(tmp_pred_y)
    pred_y = np.hstack(pred_y)
    print(classification_report(train_val_y, pred_y))


print('begin train -> ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
clf = None
num_boost_round = 3
for i in range(0, n_sampels, batch_size):
    ## this is an idea for batch training
    ## you can relpace this loop for deep learning methods
    if i % batch_size * 10 == 0:
        print("done %d/%d" % (i, n_sampels), ' -> ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    start_pos = i
    end_pos = min(i + batch_size, n_sampels)
    train_s1_batch = np.asarray(train_s1[start_pos:end_pos, :, :, :])
    train_s2_batch = np.asarray(train_s2[start_pos:end_pos, :, :, :])
    cur_batch_size = train_s2_batch.shape[0]
    train_s1_batch = train_s1_batch.reshape((cur_batch_size, -1))
    train_s2_batch = train_s2_batch.reshape((cur_batch_size, -1))
    train_X_batch = np.hstack([train_s1_batch, train_s2_batch])
    label_batch = train_y[start_pos:end_pos]
    dtrain_2class = xgb.DMatrix(train_X_batch, label=label_batch)
    print(train_X_batch.shape)
    print(label_batch.shape)
    if i == 0:
        gbdt = xgb.train(params, dtrain_2class, num_boost_round=num_boost_round)
    else:
        gbdt = xgb.train(params, dtrain_2class, num_boost_round=num_boost_round, xgb_model=clf)
    clf = gbdt
    # print(clf.get_dump())
    val(clf)
print('end train -> ', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
