from torch.utils.data import random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from dataset import DatasetLandslide
from model import LandslideModel

# Path: scripts_pl/dataset.py
data_path = '/home/ryali93/Desktop/l4s/data/TrainData'

# load dataset
dataset = DatasetLandslide(data_path)

# split dataset in train and val with random_split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# print the dataset size
print('Train dataset size: ', len(train_dataset))

# Crear los DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

def train():
    model = LandslideModel()
    wandb_logger = WandbLogger(project="m_thesis_pl", entity="ryali")
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    trainer = Trainer(max_epochs=100, logger=wandb_logger, callbacks=[checkpoint_callback, early_stop_callback]) #
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.save_checkpoint("models/model_pl1.ckpt")

# def test():
#     model = LandslideModel.load_from_checkpoint(checkpoint_path="model_pl1.ckpt")
#     test_dataloader = DataLoader(test_dataset, batch_size=16)
#     trainer = Trainer()
#     trainer.test(model, test_dataloader)

if __name__ == '__main__':
    train()
    # test()
