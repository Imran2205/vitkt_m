from __future__ import print_function
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from tcNet_torch import tclstm
import os
from tcopt import tcopts
import torch.utils.data as torch_dataset

class CustomDataset(torch_dataset.Dataset):
    def __init__(self, pos_data, neg_data):
        self.pos_data=pos_data
        self.neg_data=neg_data
        self.pos_num=len(pos_data)
        self.neg_num=len(neg_data)

    def __getitem__(self, index):
        pos_id = np.random.randint(0, self.pos_num)
        neg_id = np.random.randint(0, self.neg_num)

        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], self.pos_data[pos_id][1]), allow_pickle=True)
        frame_index = self.pos_data[pos_id][0][:, 4]
        pos_map = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])#[20,1,19,19]

        map_tmp = np.load(os.path.join(tcopts['train_data_dir'], self.neg_data[neg_id][1]), allow_pickle=True)
        frame_index = self.neg_data[neg_id][0][:, 4]
        neg_map = np.reshape(map_tmp[np.array(frame_index, dtype='int16') - 1], [tcopts['time_steps'], 1, 19, 19])


        '''TODO'''
        pos_input = self.pos_data[pos_id][0]#[20,8]
        neg_input= self.neg_data[neg_id][0]
        return pos_input,neg_input,pos_map,neg_map

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.pos_num


def load_training_data(pos_name, neg_name):
    pos_data = np.load(pos_name, allow_pickle=True)
    neg_data = np.load(neg_name, allow_pickle=True)
    return pos_data, neg_data
def prepare_test_data(Dataset, seq_len, mode=None):
    base_dir = tcopts['train_data_dir']
    data_dir = os.path.join(base_dir, Dataset)
    train_list = os.listdir(data_dir)
    train_list.sort()
    np.random.shuffle(train_list)

    testing_set_dir = '../../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        train_list = [vid for vid in train_list if vid[:-4] in testing_set]
    elif mode == 'train':
        print('train data')
        train_list = [vid for vid in train_list if vid[:-4] not in testing_set]
    else:
        print("all data")
    pos_data, neg_data = prepare_data(data_dir, seq_len, train_list)
    np.save('test_neg_data.npy', np.array(neg_data))
    np.save('test_pos_data.npy', np.array(pos_data))
    return pos_data, neg_data

def prepare_train_data(Dataset, seq_len, mode=None):
    base_dir = tcopts['train_data_dir']
    data_dir = os.path.join(base_dir, Dataset)
    train_list = os.listdir(data_dir)
    train_list.sort()
    np.random.shuffle(train_list)

    testing_set_dir = '../../utils/testing_set.txt'
    testing_set = list(np.loadtxt(testing_set_dir, dtype=str))
    if mode == 'test':
        print('test data')
        train_list = [vid for vid in train_list if vid[:-4] in testing_set and vid.endswith('.txt')]
    elif mode == 'train':
        print('train data')
        train_list = [vid for vid in train_list if vid[:-4] not in testing_set and vid.endswith('.txt')]
    else:
        print("all data")

    pos_data, neg_data = prepare_data(data_dir, seq_len,train_list)
    np.save('neg_data.npy', np.array(neg_data))
    np.save('pos_data.npy', np.array(pos_data))
    return pos_data, neg_data

def prepare_data(data_dir, seq_len, train_list):


    pos_data = []
    neg_data = []
    sampling_interval = tcopts['sampling_interval']
    # video
    for id, video in enumerate(train_list):
        print(str(id) + ':' + video)
        txt_tmp = np.loadtxt(os.path.join(data_dir, train_list[id]), delimiter=',')#[box, im_id, iou, score_max, dis]
        map_tmp = np.load(os.path.join(data_dir, train_list[id][:-4]+'_map.npy'))
        loss_list = np.where(txt_tmp[:, 5] == 0)[0]
        for i in range((len(txt_tmp) - seq_len)//sampling_interval):
            if sampling_interval * i + seq_len + 1 >= len(txt_tmp):
                continue
            data_tmp = txt_tmp[sampling_interval*i+1:sampling_interval*i + seq_len+1]
            loss_index = np.concatenate([np.where(data_tmp[:, 5] == -1)[0], np.where(data_tmp[:, 5] == 0)[0]])
            if data_tmp[-1, 5] > tcopts['pos_thr']:
                # pos data
                pos_data.append([data_tmp, train_list[id][:-4]+'_map.npy'])
            elif data_tmp[-1, 5] == 0:
                neg_data.append([data_tmp, train_list[id][:-4]+'_map.npy'])
    return pos_data, neg_data


