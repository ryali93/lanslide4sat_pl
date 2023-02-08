import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import wandb

# read functions
from dataset import read_data, config_data
from model import unet_model
from metrics import *

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# CONFIGS
path_dataset = "data_val"

# READ DATA
path_train, path_mask = read_data(path_dataset)
TRAIN_XX, TRAIN_YY = config_data(path_train, path_mask)

# SPLIT DATASET
x_train, x_valid, y_train, y_valid = train_test_split(TRAIN_XX, TRAIN_YY, test_size=0.2, shuffle=True)

# to release some memory, delete the unnecessary variable
del TRAIN_XX
del TRAIN_YY
del path_train
del path_mask

# CONFIGURATION
wandb.init(project="m_thesis_dev", entity="ryali")
config = wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 16
}

IMG_SHAPE  = (128, 128, 15)
EPOCHS = 10
model = unet_model(128, 128, 15)

def train():
    checkpointer = tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor="f1_m", verbose=1, save_best_only=True, mode="max")
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='f1_m', patience=10, verbose=1, mode='max')
    wdb = wandb.keras.WandbCallback()
    callbacks = [
        # earlyStopping,
        checkpointer,
        wdb
        ]
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', f1_m, precision_m, recall_m, dice_loss])
    history = model.fit(x_train, 
                        y_train, 
                        verbose = 2,
                        validation_data=(x_valid, y_valid),
                        epochs=EPOCHS,
                        batch_size=16,
                        callbacks=callbacks)
    
    loss_r, accuracy_r, f1_r, precision_r, recall_r, dice_r = model.evaluate(x_valid, y_valid, verbose=0)

    print(loss_r, accuracy_r, f1_r, precision_r, recall_r, dice_r)
    
    wandb.log({
        "loss": loss_r,
        "accuracy": accuracy_r,
        "fi": f1_r,
        "dice": dice_r
        })

    # model.save("models/model_564_15b.h5")
    # json.dump(history.history, open("models/history_564_15b.json", 'w'))


    wandb.log({
        'images': wandb.Image(images[0].cpu()),
        'masks': {
            'true': wandb.Image(true_masks[0].float().cpu()),
            'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
            },
        'step': global_step,
        'epoch': epoch
    })


if __name__ == '__main__':
    train()