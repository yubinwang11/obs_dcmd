"""
Created: September 2021

@author: Yubin Wang
"""

from numpy.core.fromnumeric import shape
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchsnooper
from log_normalozaition import log_transfer_torch
from scipy import io

class DCMD_Dataset(Dataset):

    x_dim, y_dim, u_dim = 10, 3, 2
    dt = .01

    #@profile
    def __init__(self, n_samples, std_x=None, cuda_button=False):
        super().__init__()
 
        X = io.loadmat('./DataSet/X_DCMD.mat').get('X').astype(np.float32)
        Y = io.loadmat('./DataSet/Y_DCMD.mat').get('Y').astype(np.float32)

        X = torch.tensor(X).to(torch.float32).cuda()
        Y = torch.tensor(Y).to(torch.float32).cuda()

        self.n_samples = n_samples

        self.x = np.empty((0, self.x_dim))
        self.x_next = np.empty((0, self.x_dim))
        self.y = np.empty((0, self.y_dim))
        self.y_next = np.empty((0, self.y_dim))

        if cuda_button:
            if torch.cuda.is_available():
                x = torch.tensor(self.x).to(torch.float32)
                self.x = x.cuda()
                y = torch.tensor(self.y).to(torch.float32)
                self.y = y.cuda()
                x_next = torch.tensor(self.x_next).to(torch.float32)
                self.x_next = x_next.cuda()
                y_next = torch.tensor(self.y_next).to(torch.float32)
                self.y_next = y_next.cuda()

        for num in range(n_samples):
            line_num = np.random.randint(0, len(X))
            invalid_num = 0
            dual_num = 0
            if (line_num+1) % 500 ==0:
                invalid_num += 1
                new_line = np.random.randint(0, len(X))
                if (new_line+1) % 500 != 0:
                    line_num = torch.tensor(new_line).to(torch.long).cuda()
                else:
                    dual_num += 1
                    dualnew_line = np.random.randint(0, len(X))
                    if (dualnew_line+1) % 500 !=0:
                        line_num = torch.tensor(dualnew_line).to(torch.long).cuda()
                    else:
                        print('training kill cuz generating x in last stae') 
                        exit(1)

            self.x = torch.cat((self.x, X[line_num, :].unsqueeze(0)), 0)
            self.x_next = torch.cat((self.x_next, X[line_num+1].unsqueeze(0)), 0)
            self.y = torch.cat((self.y, Y[line_num].unsqueeze(0)), 0)
            self.y_next = torch.cat((self.y_next, Y[line_num+1].unsqueeze(0)), 0)
        
        self.x_max = self.x.max()

        ### data nomalization
        self.x, self.x_next = log_transfer_torch(self.x), log_transfer_torch(self.x_next)
        self.y, self.y_next = log_transfer_torch(self.y), log_transfer_torch(self.y_next)

        if std_x is None:
            self.std_x = self.x.std()
        
    def __len__(self):
        return self.n_samples

    #@profile
    def __getitem__(self, index):
        x = self.x[index, :] #/ self.std_x
        y = self.y[index, :]
        x_next = self.x_next[index, :] #/ self.std_x
        y_next = self.y_next[index, :]

        return x, y, x_next, y_next
        
####################################
############ Start Test ############

#train_dataset = DCMD_Dataset(cuda_button=True, n_samples=20000)
#print(train_dataset.data)
#valid_dataset = Valid_Dataset(cuda_button=True)

