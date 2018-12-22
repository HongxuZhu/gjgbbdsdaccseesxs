import os
import time

import h5py
import numpy as np

### to change according to your machine
base_dir = os.path.expanduser("/home/utopia/CVDATA/German_AI_Challenge_2018/session1")
path_training = os.path.join(base_dir, 'training.h5')
path_validation = os.path.join(base_dir, 'validation.h5')
path_s18_train = os.path.join(base_dir, 's18_train.h5')
path_s18_val = os.path.join(base_dir, 's18_val.h5')
path_s2_train_index = os.path.join(base_dir, 's2_train_index.h5')
path_s2_val_index = os.path.join(base_dir, 's2_val_index.h5')

fid_training = h5py.File(path_training, 'r')
fid_validation = h5py.File(path_validation, 'r')

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


def save_h5_batch(rfiles, wfile, start_pos, end_pos, total, dataname, wflag):
    s1_batch = rfiles['s1'][start_pos:end_pos, :, :, :]
    s2_batch = rfiles['s2'][start_pos:end_pos, :, :, :]
    hstack = np.concatenate((s1_batch, s2_batch), axis=3)
    if wflag:
        h5f = h5py.File(wfile, 'w')
        h5f.create_dataset(dataname, (total, 32, 32, 18),
                           # maxshape=(None, 32, 32, 18),
                           # chunks=(1, 1000, 1000),
                           dtype='float32')
    else:
        h5f = h5py.File(wfile, 'a')
    dataset = h5f[dataname]
    dataset[start_pos:end_pos] = hstack
    print(dataname, start_pos, end_pos, wflag, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    h5f.close()


def save_h5(hstack_size, train_n_sampels, rfiles, wfile, dataname):
    for i in range(0, train_n_sampels, hstack_size):
        start_pos = i
        end_pos = min(i + hstack_size, train_n_sampels)
        save_h5_batch(rfiles, wfile, start_pos, end_pos, train_n_sampels, dataname, i == 0)


def save():
    hstack_size = 30000
    rfiles_train = {'s1': s1_training, 's2': s2_training}
    rfiles_val = {'s1': s1_validation, 's2': s2_validation}

    save_h5(hstack_size, 352366, rfiles_train, path_s18_train, 's18_train')
    save_h5(hstack_size, 24119, rfiles_val, path_s18_val, 's18_val')
    fid_training.close()
    fid_validation.close()


def check(i, path, dataset, s1_orig, s2_orig):
    j = i + 1
    s18_train = h5py.File(path, 'r')
    s1 = s1_orig[i:j, :, :, :]
    s2 = s2_orig[i:j, :, :, :]
    hstack = np.concatenate((s1, s2), axis=3)
    s18 = s18_train[dataset][i:j, :, :, :]
    print(hstack.shape, hstack[0][9][9])
    print()
    print(s18.shape, s18[0][9][9])
    print()


# B2 B3 B4 B5 B6 B7 B8 B8A B11 B12
# 0  1  2  3  4  5  6  7   8   9
def index_s2(start_pos, end_pos, s2_orig, wfile, dataname, total, wflag):
    s2 = s2_orig[start_pos:end_pos:, :, :]
    B4 = s2[:, :, :, 2:3]
    # B5 = s2[:, :, :, 3:4]
    # B7 = s2[:, :, :, 5:6]
    B8 = s2[:, :, :, 6:7]
    B11 = s2[:, :, :, 8:9]
    # C1_Red_edge = np.divide(B7, B5) - 1  # RuntimeWarning: divide by zero encountered in true_divide
    NDVI = np.divide(np.subtract(B8, B4), np.add(B8, B4))
    NDWI = np.divide(np.subtract(B8, B11), np.add(B8, B11))
    EVI2 = 2.5 * np.divide(np.subtract(B8, B4), np.add(B8, 2.4 * B4) + 1)
    hstack = np.concatenate((NDVI, NDWI, EVI2), axis=3)
    if wflag:
        h5f = h5py.File(wfile, 'w')
        h5f.create_dataset(dataname, (total, 32, 32, 3),
                           dtype='float32')
    else:
        h5f = h5py.File(wfile, 'a')
    dataset = h5f[dataname]
    dataset[start_pos:end_pos] = hstack
    print(dataname, start_pos, end_pos, wflag, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    h5f.close()


def index(hstack_size, train_n_sampels, rfiles, wfile, dataname):
    for i in range(0, train_n_sampels, hstack_size):
        start_pos = i
        end_pos = min(i + hstack_size, train_n_sampels)
        index_s2(start_pos, end_pos, rfiles, wfile, dataname, train_n_sampels, i == 0)


def dist():
    import matplotlib.pyplot as plt

    train_class_cnt = label_training[:, :].sum(0) / 3523.66
    val_class_cnt = label_validation[:, :].sum(0) / 241.19
    chaju = train_class_cnt / val_class_cnt - 1
    print(train_class_cnt)
    print(val_class_cnt)
    print(chaju)

    plt.subplot(1, 3, 1)
    plt.barh(range(17), train_class_cnt)
    plt.yticks(range(17))
    plt.xlabel("freq")
    plt.ylabel("class")
    plt.title("train")
    # for x, y in enumerate(train_class_cnt):
    #     plt.text(y + 0.2, x - 0.1, '%s' % y)

    plt.subplot(1, 3, 2)
    plt.barh(range(17), val_class_cnt)
    plt.yticks(range(17))
    plt.xlabel("freq")
    plt.ylabel("class")
    plt.title("val")

    plt.subplot(1, 3, 3)
    plt.barh(range(17), chaju)
    plt.yticks(range(17))
    plt.xlabel("freq")
    plt.ylabel("class")
    plt.title("train/val - 1")
    plt.show()


# save()
# check(352365, path_s18_train, 's18_train', s1_training, s2_training)
# check(2365, path_s18_val, 's18_val', s1_validation, s2_validation)
# dist()

index(30000, 352366, s2_training, path_s2_train_index, 's2_train_index')
index(30000, 24119, s2_validation, path_s2_val_index, 's2_val_index')
