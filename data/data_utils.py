import os
#import cv2
import numpy as np
#import skimage.io
import random
import scipy.io as sio


def convert_mat_to_list(mat, dataset, convert_type='simple'):
    ret_list = []
    if dataset == 'cub':
        for i in mat:
            ret_list.append(os.path.basename(i[0][0]))
    elif dataset == 'flo':
        for i in mat:
            ret_list.append(os.path.basename(i[0][0]))
    elif dataset == 'sun':
        if convert_type == 'simple':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/')[8:])))
        elif convert_type == 'origin':
            for i in mat:
                ret_list.append(str(i[0][0]))
                # import pdb; pdb.set_trace()
    elif dataset == 'awa2':
        if convert_type == 'origin':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/'))))
        elif convert_type == 'simple':
            for i in mat:
                ret_list.append(str("/".join(i[0][0].split('/')[-2:])))
    elif dataset == 'apy':
        if convert_type == 'simple':
            for i in mat:
                i_part = i[0][0].split('/')
                if len(i_part) == 12: # VOC image
                    ret_list.append(str("/".join(i[0][0].split('/')[8:])))
                elif len(i_part) == 9: #aYahoo image
                    ret_list.append(str("/".join(i[0][0].split('/')[-2:])))
                else:
                    raise NotImplementedError
    else:
        raise NotImplementedError

    return ret_list

    
def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath

def checkdir(datapath):
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def norm_feat(data):
    """
    e.g. CUB class attr size: 200 x 312
    """
    assert data.ndim == 2
    data_len = np.linalg.norm(data, axis=1)
    data_len += 1e-8
    norm_data = data / data_len[:, None]
    return norm_data


if __name__ == '__main__':
    res_mat = sio.loadmat('D:/ZSL-Dataset/xlsa17/data/AWA2/res101.mat')
    split_mat = sio.loadmat('D:/ZSL-Dataset/xlsa17/data/AWA2/att_splits.mat')
    all_att = split_mat['att'][...]
    image_files = res_mat['image_files'][...]
    image_labels = res_mat['labels'][...] - 1
    ret_list = convert_mat_to_list(image_files, 'awa2', 'simple')
    trainval_loc = split_mat['trainval_loc'][...] - 1
    test_seen_loc = split_mat['test_seen_loc'][...] - 1
    test_unseen_loc = split_mat['test_unseen_loc'][...] - 1

    # train_files = np.array(ret_list)[trainval_loc]
    # train_labels = image_labels[trainval_loc]
    # Note = open('./AWA2/train.list', 'w')
    # for i, j in zip(train_files, train_labels):
    #     Note.write(i[0])
    #     Note.write(' ')
    #     Note.write(str(j[0][0]))
    #     Note.write('\n')

    import h5py
    with h5py.File('./AWA2/data_info.h5', mode='w') as f:
        f.create_dataset('all_att', data=all_att.T)
        f.create_dataset('seen_class', data=np.unique(image_labels[trainval_loc]))
        f.create_dataset('unseen_class', data=np.unique(image_labels[test_unseen_loc]))



