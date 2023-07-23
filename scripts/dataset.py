import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import h5py

norm_dicc = {
    "l4s": {
        "B1": [0.0, 3.1057398816745447],
        "B2": [0.0, 19.76154895369598],
        "B3": [0.0, 31.551829580344585],
        "B4": [0.0, 33.16014987560649],
        "B5": [0.0, 9.972775876378055],
        "B6": [0.0, 4.144120061411549],
        "B7": [0.0, 3.6925308568047677],
        "B8": [0.0, 8.312890666119067],
        "B9": [0.0, 3.5485627183025263],
        "B10": [0.0, 21.44150144339325],
        "B11": [0.0, 5.786441765686274],
        "B12": [0.0, 19.661322873426624],
        "B13": [0.0, 4.040945090994509],
        "B14": [0.0, 5.121427202344607]
    },
    "own": {
        "B1": [681.0, 3165.0],
        "B2": [183.0, 8655.0],
        "B3": [258.0, 7697.0],
        "B4": [68.0, 6779.0],
        "B5": [1.0, 5293.0],
        "B6": [0.0, 5600.0],
        "B7": [58.0, 6544.0],
        "B8": [111.0, 8930.0],
        "B9": [36.0, 2972.0],
        "B10": [1.0, 270.0],
        "B11": [36.0, 7280.0],
        "B12": [1.0, 12937.0],
        "B13": [0.0, 1.4837092161178589],
        "B14": [0.000001, 4128.0]
    }
}

def normalize_minmax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# class DatasetLandslide(Dataset):
#     def __init__(self, path):
#         self.train_paths, self.mask_paths = self.read_data(path)
#         self.train_data, self.mask_data = self.config_data(self.train_paths, self.mask_paths)

#     def read_data(self, path):
#         TRAIN_PATH = f"{path}/img/*.h5" # data_val
#         TRAIN_MASK = f'{path}/mask/*.h5' # data_val
#         all_train = sorted(glob.glob(TRAIN_PATH))
#         all_mask = sorted(glob.glob(TRAIN_MASK))
#         return all_train, all_mask

#     def config_data(self, all_train, all_mask):
#         n_imgs = len(all_train)
#         TRAIN_XX = np.zeros((n_imgs, 14, 128, 128)) # PyTorch utiliza el formato (N, C, H, W)
#         TRAIN_YY = np.zeros((n_imgs, 1, 128, 128))  # PyTorch utiliza el formato (N, C, H, W)
#         for i, (img, mask) in enumerate(zip(all_train, all_mask)):
#             with h5py.File(img) as hdf:
#                 data = np.array(hdf.get('img'))
#                 data[np.isnan(data)] = 0.000001
#                 TRAIN_XX[i, :, :, :] = data.transpose((2, 0, 1)) # Transponemos para tener (C, H, W)
    
#             with h5py.File(mask) as hdf:
#                 data=np.array(hdf.get('mask'))
#                 TRAIN_YY[i, :, :] = data

#         TRAIN_XX[np.isnan(TRAIN_XX)] = 0.000001
#         return torch.from_numpy(TRAIN_XX).float(), torch.from_numpy(TRAIN_YY).float() # Convertimos a tensores de PyTorch

#     def __len__(self):
#         return len(self.train_data)

#     def __getitem__(self, idx):
#         return self.train_data[idx], self.mask_data[idx]

class DatasetLandslide(Dataset):
    def __init__(self, path):
        self.train_paths, self.mask_paths = self.read_data(path)

    def read_data(self, path):
        TRAIN_PATH = f"{path}/img/*.h5" # data_val
        TRAIN_MASK = f'{path}/mask/*.h5' # data_val
        all_train = sorted(glob.glob(TRAIN_PATH))
        all_mask = sorted(glob.glob(TRAIN_MASK))
        return all_train, all_mask

    def __len__(self):
        return len(self.train_paths)

    def __getitem__(self, idx):
        with h5py.File(self.train_paths[idx]) as hdf:
            data = np.array(hdf.get('img'))
            data[np.isnan(data)] = 0.000001
            img = data.transpose((2, 0, 1))  # Transponemos para tener (C, H, W)
            img = normalize_minmax(img)
            # if "image_" in self.train_paths[idx]:
            #     img = normalize_minmax(img)
            # else:
            #     img = img[0:3]

        with h5py.File(self.mask_paths[idx]) as hdf:
            mask = np.array(hdf.get('mask'))
            mask = mask[np.newaxis, :]  # Añadir dimensión de canal

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()  # Convertimos a tensores de PyTorch
