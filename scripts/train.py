import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from dataset import DatasetLandslide
from model import LandslideModel
from utils import *
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

set_seed(config["train_config"]["seed"])

# Path: scripts_pl/dataset.py
data_path = config["train_config"]["dataset_path"]

# load dataset
dataset = DatasetLandslide(data_path)

# split dataset in train and val with random_split
train_val_split = config["train_config"]["train_val_split"]
train_size = int(train_val_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print the dataset size
print('Train dataset size: ', len(train_dataset))

# Crear los DataLoaders
train_dataloader = DataLoader(train_dataset, 
                              batch_size = config["train_config"]["batch_size"], 
                              shuffle = True, 
                              worker_init_fn = worker_init_fn)
val_dataloader = DataLoader(val_dataset, 
                            batch_size = config["train_config"]["batch_size"], 
                            worker_init_fn = worker_init_fn)

def train():
    """
    Trains a LandslideModel using the provided train and validation dataloaders.
    Uses WandbLogger for logging and ModelCheckpoint and EarlyStopping for callbacks.
    Saves the best model checkpoint in both .ckpt and .pth formats.
    """
    model = LandslideModel()
    wandb_logger = WandbLogger(project="lanslide4sat_pl", entity="ryali")
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", 
                                          mode="min", 
                                          save_top_k=1)
    early_stop_callback = EarlyStopping(monitor='val_loss', 
                                        patience=10, 
                                        mode='min')
    trainer = Trainer(max_epochs=100, 
                      logger=wandb_logger, 
                      callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("models/modelo_test.ckpt")

if __name__ == '__main__':
        train()