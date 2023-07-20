import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import h5py

class DatasetLandslide(Dataset):
    def __init__(self, path):
        self.train_paths, self.mask_paths = self.read_data(path)
        self.train_data, self.mask_data = self.config_data(self.train_paths, self.mask_paths)

    def read_data(self, path):
        TRAIN_PATH = f"{path}/img/*.h5" # data_val
        TRAIN_MASK = f'{path}/mask/*.h5' # data_val
        all_train = sorted(glob.glob(TRAIN_PATH))
        all_mask = sorted(glob.glob(TRAIN_MASK))
        return all_train, all_mask

    def config_data(self, all_train, all_mask):
        n_imgs = len(all_train)
        TRAIN_XX = np.zeros((n_imgs, 14, 128, 128)) # PyTorch utiliza el formato (N, C, H, W)
        TRAIN_YY = np.zeros((n_imgs, 1, 128, 128)) # PyTorch utiliza el formato (N, C, H, W)
        for i, (img, mask) in enumerate(zip(all_train, all_mask)):
            with h5py.File(img) as hdf:
                data = np.array(hdf.get('img'))
                data[np.isnan(data)] = 0.000001
                TRAIN_XX[i, :, :, :] = data.transpose((2, 0, 1)) # Transponemos para tener (C, H, W)
    
            with h5py.File(mask) as hdf:
                data=np.array(hdf.get('mask'))
                TRAIN_YY[i, :, :] = data

        TRAIN_XX[np.isnan(TRAIN_XX)] = 0.000001
        return torch.from_numpy(TRAIN_XX).float(), torch.from_numpy(TRAIN_YY).float() # Convertimos a tensores de PyTorch

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        return self.train_data[idx], self.mask_data[idx]
