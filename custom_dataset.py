from torchvision.datasets.folder import default_loader
import os
from torch.utils.data.dataset import Dataset
import torch
import scipy.io as sio
import h5py
import numpy as np
from sklearn.decomposition import PCA


class MyCustomDataset(Dataset):
    def __init__(self, dataset='wiki_shallow', state='train'):

        if dataset == 'pascal_deep':
            data_dir = './datasets/extracted_feature/'
            train_img = sio.loadmat(data_dir + 'train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            T_train = train_txt['train_txt']

            test_img = sio.loadmat(data_dir + 'test_img.mat')
            I_test = test_img['test_img']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            labels_train = train_img_lab['train_img_lab']
            test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            labels_test = test_img_lab['test_img_lab']


        elif dataset == 'xmedianet_deep':
            data_dir = './datasets/extracted_feature/'
            train_img = sio.loadmat(data_dir + 'train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            T_train = train_txt['train_txt']

            test_img = sio.loadmat(data_dir + 'test_img.mat')
            I_test = test_img['test_img']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            labels_train = train_img_lab['train_img_lab']
            test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            labels_test = test_img_lab['test_img_lab']


        elif dataset == 'nus_deep':
            data_dir = './datasets/extracted_feature/'
            train_img = sio.loadmat(data_dir + 'train_img.mat')
            I_train = train_img['train_img']
            train_txt = sio.loadmat(data_dir + 'train_txt.mat')
            T_train = train_txt['train_txt']

            test_img = sio.loadmat(data_dir + 'test_img.mat')
            I_test = test_img['test_img']
            test_txt = sio.loadmat(data_dir + 'test_txt.mat')
            T_test = test_txt['test_txt']

            train_img_lab = sio.loadmat(data_dir + 'train_img_lab.mat')
            labels_train = train_img_lab['train_img_lab']
            test_img_lab = sio.loadmat(data_dir + 'test_img_lab.mat')
            labels_test = test_img_lab['test_img_lab']

        if state == 'train':
            self.I = I_train
            self.T = T_train
            self.labels = labels_train

        if state == 'test':
            self.I = I_test
            self.T = T_test
            self.labels = labels_test

        self.I = torch.FloatTensor(self.I)

        if dataset == 'wiki_deep' or dataset == 'pascal_deep' \
                or dataset == 'xmedianet_deep' or dataset == 'nus_deep':
            self.T = torch.FloatTensor(self.T)
        elif dataset == 'xmedianet' or dataset == 'nus_deep' or dataset == 'wiki_deep_corr-ae':
            self.T = torch.LongTensor(self.T)

        self.labels = torch.LongTensor(self.labels)
        self.labels = self.labels.view(-1, 1)

    def __getitem__(self, index):
        I_item, T_item, label = self.I[index], self.T[index], self.labels[index]
        return I_item, T_item, label

    def __len__(self):
        count = len(self.I)
        # print (len(self.I), len(self.T), len(self.labels))
        assert len(self.I) == len(self.T) == len(self.labels)
        return count
