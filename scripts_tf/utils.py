from pathlib import Path
import h5py
import numpy as np
import os

def valid_images(list_img: list, list_mask: list):
    return list_img, list_mask

def list_images(folder_path: Path):
    list_img = os.listdir(folder_path / "img")
    list_mask = os.listdir(folder_path / "mask")
    valid_images(list_img, list_mask)
    return list_img, list_mask

def read_h5(img_path: Path):
    with h5py.File(img_path) as hdf:
        ls = list(hdf.keys())
        data = np.array(hdf.get("img"))
    return data
