import pandas as pd
import numpy as np
from pathlib import Path
from pathlib import PurePath
from scipy.stats.stats import pearsonr
from os.path import join


from torch.utils.data import Dataset, DataLoader
import torch



def get_data_loader(config, data, shuffle=False, pin_memory=False, drop_last=False):
    dataset = Dataset(data_info=data, model=config.model, cway=config.classification_way, 
                      folder_name=config.folder_name, transform='ToTensor')
    data_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers, 
                             shuffle=shuffle, pin_memory=pin_memory, drop_last=drop_last)
    num_data = len(dataset)

    return (data_loader, num_data)


class Dataset(torch.utils.data.Dataset): 
    def __init__(self, data_info, model, cway, folder_name, transform=None):
        self.data_info = data_info
        self.model = model
        self.cway = cway
        self.transform = transform
        self.data_path = f'./data/{folder_name}'

    def __len__(self):
        return self.data_info.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        subject_id, diagnosis, label = self.data_info.iloc[idx].subject_id, self.data_info.iloc[idx].diagnosis, self.data_info.iloc[idx].label
        
        fnc_dir = join(self.data_path, subject_id+'.npy')
        sFNC = np.load(fnc_dir).astype('float32')
        if self.model=='FNN':
            sFNC = sFNC[np.triu_indices(53, k = 1)] # vectorized upper triangle of sFNC
        else:
            sFNC = np.expand_dims(sFNC,axis=0)
        
        if isinstance(self.transform, str):
            self.transform = globals()[self.transform]()

        sFNC = self.transform(sFNC)
            
        return sFNC, label
    
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, x):
        # normalization
        Min, Max = np.amin(x), np.amax(x)
        x = (x - Min) / (Max - Min)

        return torch.from_numpy(x)
    
    
    