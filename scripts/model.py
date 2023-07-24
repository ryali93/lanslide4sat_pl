import pytorch_lightning as pl
from torch import nn, sigmoid
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torchmetrics
import segmentation_models_pytorch as smp

import wandb

class_labels = {0: "no landslide", 1: "landslide"}
segmentation_classes = ["nld", "ld"]
def labels():
    l = {}
    for i, label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wandb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction" : {
            "mask_data" : pred_mask, 
            "class_labels" : labels()
        },
        "ground truth" : {
            "mask_data" : true_mask, 
            "class_labels" : labels()
        }})

class unet_model(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(unet_model, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1)
        self.model = smp.Unet(
            encoder_name="mit_b0",        # resnet34 # choose encoder, e.g. mobilenet_v2 | efficientnet-b7 | mobilenet_v2
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=3,        # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=num_classes,            # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.model(x)

class fpn_model(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(fpn_model, self).__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1) # Convierte de 14 a 3 canales usando un kernel de tamaño 1

        self.model = smp.FPN(
            encoder_name="mit_b2",          # 
            encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
            in_channels=3,        # model input channels (1 for grayscale images, 3 for RGB, etc.)
            classes=num_classes,            # model output channels (number of classes in your dataset)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.model(x)

class LandslideModel(pl.LightningModule):
    def __init__(self):
        super(LandslideModel, self).__init__()
        # unet_model(128, 128, 15) # Asume que unet_model devuelve una instancia de un modelo PyTorch
        # self.model = unet_model(14, 1, 1)
        self.model = unet_model(6, 1, 1)
        self.criterion = nn.BCEWithLogitsLoss() # Asume que estás usando la pérdida de entropía cruzada binaria

        self.train_f1 = torchmetrics.F1Score(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')

        self.train_precision = torchmetrics.Precision(task='binary')
        self.val_precision = torchmetrics.Precision(task='binary')

        self.train_iou = torchmetrics.JaccardIndex(task='binary')
        self.val_iou = torchmetrics.JaccardIndex(task='binary')

        # self.train_dice_loss = torchmetrics.Dice(num_classes=2)
        # self.val_dice_loss = torchmetrics.Dice(num_classes=2)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        precision = self.train_precision(y_hat, y)
        iou = self.train_iou(y_hat, y)
        # loss_dice = self.train_dice_loss(y_hat, y)
        loss = self.criterion(y_hat, y)
        loss_f1 = self.train_f1(y_hat, y)
        self.log('train_precision', precision)
        self.log('train_iou', iou)
        # self.log('train_dice_loss', loss_dice)
        self.log('train_loss', loss)
        self.log('train_f1', loss_f1)

        # Loguear imágenes en wandb
        if self.current_epoch % 10 == 0:  # Loguear cada 10 épocas
            x = x[:, 1:4]
            x = x.permute(0, 2, 3, 1)

            # Loguear las primeras imágenes del lote
            self.logger.experiment.log({
                "image": wandb.Image(x[0].cpu().detach().numpy()*255, masks={
                    "predictions": {
                        "mask_data": y_hat[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": y[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    }
                })
            })

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        precision = self.val_precision(y_hat, y)
        iou = self.val_iou(y_hat, y)
        # loss_dice = self.val_dice_loss(y_hat, y)
        loss = self.criterion(y_hat, y)
        loss_f1 = self.val_f1(y_hat, y)
        self.log('val_precision', precision)
        self.log('val_iou', iou)
        # self.log('val_dice_loss', loss_dice)
        self.log('val_loss', loss)
        self.log('val_f1', loss_f1)

        # Loguear imágenes en wandb
        if self.current_epoch % 10 == 0:  # Loguear cada 10 épocas
            x = x[:, 1:4]
            x = x.permute(0, 2, 3, 1)

            # Loguear las primeras imágenes del lote
            self.logger.experiment.log({
                "image": wandb.Image(x[0].cpu().detach().numpy()*255, masks={
                    "predictions": {
                        "mask_data": y_hat[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    },
                    "ground_truth": {
                        "mask_data": y[0][0].cpu().detach().numpy(),
                        "class_labels": class_labels
                    }
                })
            })
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)
