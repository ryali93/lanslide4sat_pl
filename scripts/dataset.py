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
        TRAIN_XX = np.zeros((128, 128, 6))
        with h5py.File(self.train_paths[idx]) as hdf:
            data = np.array(hdf.get('img'))
            data[np.isnan(data)] = 0.000001
            
            # data_red = data[:, :, 3]
            # data_nir = data[:, :, 7]
            # data_ndvi = np.divide(data_nir - data_red, np.add(data_nir, data_red))

            TRAIN_XX[:, :, 0] = data[:, :, 1]
            TRAIN_XX[:, :, 1] = data[:, :, 2]
            TRAIN_XX[:, :, 2] = data[:, :, 3]
            TRAIN_XX[:, :, 3] = data[:, :, 4]
            TRAIN_XX[:, :, 4] = data[:, :, 12]
            TRAIN_XX[:, :, 5] = data[:, :, 13]
            # TRAIN_XX[:, :, 6] = data_ndvi

            img = TRAIN_XX.transpose((2, 0, 1))  # Transponemos para tener (C, H, W)
            img = normalize_minmax(img)

        with h5py.File(self.mask_paths[idx]) as hdf:
            mask = np.array(hdf.get('mask'))
            mask = mask[np.newaxis, :]  # Añadir dimensión de canal

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()  # Convertimos a tensores de PyTorch
