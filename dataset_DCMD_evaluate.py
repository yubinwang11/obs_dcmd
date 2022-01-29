"""
Created: September 2021

@author: Yubin Wang
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from log_normalozaition import log_transfer_numpy, log_recover
from scipy import io

class Evaluate_Dataset(Dataset):

    x_dim, y_dim, u_dim = 10, 3, 2
    dt = .01

    def __init__(self, simu_len, std_x=None):
        super().__init__()
 
        X = io.loadmat('./DataSet/X_DCMD.mat').get('X').astype(np.float32)
        Y = io.loadmat('./DataSet/Y_DCMD.mat').get('Y').astype(np.float32)

        self.simu_len = simu_len

        self.x = np.empty((0, self.x_dim))
        self.y = np.empty((0, self.y_dim))

        line_first = np.random.randint(0, 101*101)
        first_num = line_first*500

        if self.simu_len > 500:

            for i in range(500):
                self.x = np.row_stack((self.x, X[first_num+i]))
                self.y = np.row_stack((self.y, Y[first_num+i]))
            
            for i in range(self.simu_len-500):
                self.x = np.row_stack((self.x, X[first_num+499]))
                self.y = np.row_stack((self.y, Y[first_num+499]))

        else:
            for i in range(self.simu_len):
                self.x = np.row_stack((self.x, X[first_num+i]))
                self.y = np.row_stack((self.y, Y[first_num+i]))

        self.x_max = self.x.max()
        
        ### data nomalization
        self.x = log_transfer_numpy(self.x)
        self.y = log_transfer_numpy(self.y)

        if std_x is None:
            self.std_x = self.x.std()
        
        self.std_x = std_x

    def __len__(self):
        return self.simu_len

    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index, :]
        return x, y
        
