import numpy as np
import torch

def log_transfer_numpy(value):
    new_value = np.log10(value ) / np.log10(value.max())
    
    return new_value

def log_transfer_torch(value):
    new_value = torch.log10(value ) / torch.log10(value.max())
    
    return new_value

def log_recover(new_value, value_max):

    value = 10 ** (new_value * np.log10(value_max))

    return value
     
