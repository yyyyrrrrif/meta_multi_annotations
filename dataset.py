#-*- coding : utf-8 -*-
# coding: utf-8
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data

### dataset
DATA_PATH = '../Downloads/LabelMe/prepared/'
N_CLASSES = 8

class LabelMe_dataset(data.Dataset):
    def __init__(self, data_type, data_path):
        self.data_type = data_type
        self.data_path = data_path
        # load train data
        # images processed by VGG16
        self.data_vgg16 = np.load(self.data_path + "data_%s_vgg16.npy" % self.data_type)
        # ground truth labels
        self.labels = np.load(self.data_path + "labels_%s.npy" % self.data_type)
        
        # data from Amazon Mechanical Turk       
        if self.data_type == 'train':
            self.answers = np.load(self.data_path + "answers.npy")

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        ftrs = self.data_vgg16[idx, :, :, :]
        true_label = self.labels[idx]

        if self.data_type == 'train':
            multi_label = self.answers[idx, :]
            return ftrs, true_label, multi_label
        else:
            return ftrs, true_label

  

