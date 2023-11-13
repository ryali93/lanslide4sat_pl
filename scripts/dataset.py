import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import h5py
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def normalize_minmax(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class DatasetLandslide(Dataset):
    def __init__(self, path):
        """
        Initializes the DatasetLandslide class.
        Args:
        - path (str): The path to the dataset.
        """
        self.train_paths, self.mask_paths = self.read_data(path)

    def read_data(self, path):
        """
        Reads the data from the given path.
        Args:
        - path (str): The path to the dataset.
        Returns:
        - all_train (list): A list of paths to the training data.
        - all_mask (list): A list of paths to the mask data.
        """
        # TRAIN_PATH = f"{path}/img/image*.h5" # data_val
        # TRAIN_MASK = f'{path}/mask/mask*.h5' # data_val
        TRAIN_PATH = f"{path}/img/*.h5" # data_val
        TRAIN_MASK = f'{path}/mask/*.h5' # data_val
        all_train = sorted(glob.glob(TRAIN_PATH))
        all_mask = sorted(glob.glob(TRAIN_MASK))
        return all_train, all_mask

    def __len__(self):
        """
        Returns the length of the dataset.

        Args:
        - None

        Returns:
        - len (int): The length of the dataset.
        """
        return len(self.train_paths)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.

        Args:
        - idx (int): The index of the item to return.

        Returns:
        - img (torch.Tensor): The image data as a PyTorch tensor.
        - mask (torch.Tensor): The mask data as a PyTorch tensor.
        """
        channels = config["dataset_config"]["channels"]
        TRAIN_XX = np.zeros((128, 128, len(channels)))

        with h5py.File(self.train_paths[idx]) as hdf:
            data = np.array(hdf.get('img'))
            data[np.isnan(data)] = 0.000001

            for i, channel in enumerate(channels):
                TRAIN_XX[:, :, i] = data[:, :, channel]

            img = TRAIN_XX.transpose((2, 0, 1))  # Transponemos para tener (C, H, W)
            if config["dataset_config"]["normalize"]:
                img = normalize_minmax(img)

        mask = np.array([])
        if self.mask_paths != []:
            with h5py.File(self.mask_paths[idx]) as hdf:
                mask = np.array(hdf.get('mask'))
                mask = mask[np.newaxis, :]  # Añadir dimensión de canal

        return torch.from_numpy(img).float(), torch.from_numpy(mask).float()  # Convertimos a tensores de PyTorch


class DatasetLandslideEval(Dataset):
    def __init__(self, patches):
        """
        Initializes the DatasetLandslide class.
        Args:
        - patches (list): A list of image patches.
        - mascaras (list, optional): A list of corresponding mask patches. Default is None.
        """
        self.patches = patches

    def __len__(self):
        """
        Returns the length of the dataset.
        Returns:
        - len (int): The length of the dataset.
        """
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        Args:
        - idx (int): The index of the item to return.
        Returns:
        - img (torch.Tensor): The image data as a PyTorch tensor.
        - mask (torch.Tensor): The mask data as a PyTorch tensor.
        """
        img = self.patches[idx].astype(np.float32)
        if config["dataset_config"]["normalize"]:
            img = normalize_minmax(img)
        return torch.from_numpy(img)#, torch.from_numpy(mask)