import json
import torch
import gcsfs
import numpy as np
import tensorflow as tf
from pprint import pprint
from google.cloud import storage
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataset import DatasetLandslide
from model import *
from ee_utils import *

def get_model_pl(model_path):
    model = LandslideModel()
    return  model.load_from_checkpoint(checkpoint_path=model_path)

class TFRecordDataset(Dataset):
    def __init__(self, fileName, side, bands):
        self.fileName = fileName
        self.side = side
        self.bands = bands
        self.featuresDict = {x: tf.io.FixedLenFeature([side, side], dtype=tf.float32) for x in bands}
        
        # Contar el número total de registros en el archivo TFRecord
        self.num_records = sum(1 for _ in tf.data.TFRecordDataset(self.fileName, compression_type='GZIP'))

    def __len__(self):
        return self.num_records

    def __getitem__(self, idx):
        # Crear un iterador que salta hasta el registro deseado y toma uno
        raw_dataset = tf.data.TFRecordDataset(self.fileName, compression_type='GZIP')
        raw_dataset = raw_dataset.skip(idx)
        example = next(iter(raw_dataset.take(1)))
        
        parsed_features = tf.io.parse_single_example(example, self.featuresDict)
        
        # Apilar imágenes
        nfeat = tf.transpose(tf.squeeze(tf.stack(list(parsed_features.values()))))
        nfeat = tf.cast(nfeat, tf.float32)
        nfeat = nfeat.numpy()

        nfeat = np.moveaxis(nfeat, -1, 0)
        # normalize minmax
        nfeat = (nfeat - np.min(nfeat)) / (np.max(nfeat) - np.min(nfeat))
        
        return torch.from_numpy(nfeat)

fileNames = ["/home/ryali93/Downloads/tesis2_ld_14b.tfrecord.gz"]
side = 128
bands = ["B" + str(x) for x in range(1, 15)]

model_path = r'/home/ryali93/Desktop/l4s/models/unet_resnet34_14b_l4s.pth'
model = get_model_pl(model_path)

dataset = TFRecordDataset(fileNames, side, bands)
dataset_ldr = DataLoader(dataset, batch_size=16)

print("dataset_ldr")

all_predictions = []
model.eval()
for batch in dataset_ldr:
    with torch.no_grad():
        predictions = model(batch)
    all_predictions.append(predictions)
all_predictions = torch.cat(all_predictions)

print(all_predictions)

# Save predictions in numpy array
np.save('/home/ryali93/Desktop/l4s/models/predictions2.npy', all_predictions.numpy())