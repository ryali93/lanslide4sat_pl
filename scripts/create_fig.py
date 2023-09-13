import torch
from torch.utils.data import DataLoader
from dataset import DatasetLandslide
from model import *
import matplotlib.pyplot as plt

model_name = "unet_resnet34_14b_full_2"

model_path = r'/home/ryali93/Desktop/l4s/models/{}.ckpt'.format(model_name)
model = LandslideModel()
model = LandslideModel.load_from_checkpoint(checkpoint_path=model_path, strict=False)

data_path = '/home/ryali93/Desktop/l4s/data/TrainData'
train_dataloader = DatasetLandslide(data_path)
train_loader = DataLoader(train_dataloader, batch_size=16, shuffle=False)

# 2. Iterar sobre el DataLoader y hacer predicciones
model.eval()  # Poner el modelo en modo evaluación
all_predictions = []
all_images = []
all_gt = []
i = 0

with torch.no_grad():
    for images, _ in train_loader:  # Aquí, estamos desempaquetando las imágenes y las máscaras, pero solo usamos las imágenes
        all_images.append(images)
        all_gt.append(_)
        predictions = model(images)
        predictions = predictions.cpu().numpy()
        all_predictions.append(predictions)
        i += 1
        if i == 10:
            break

# Crear una gráfica con la imagen, el ground truth, y la predicción para la primera imagen en el lote, para cada modelo
fig, ax = plt.subplots(3, 10, figsize=(20, 6))

for i in range(10):
    ax[0, i].imshow(all_images[0][i][1:4].permute(1, 2, 0)*1.5)
    ax[1, i].imshow(all_predictions[0][i][0] > 0.5)
    ax[2, i].imshow(all_gt[0][i][0])

    # remove axis
    ax[0, i].set_xticks([])
    ax[0, i].set_yticks([])
    ax[1, i].set_xticks([])
    ax[1, i].set_yticks([])
    ax[2, i].set_xticks([])
    ax[2, i].set_yticks([])

fig.subplots_adjust(hspace=0.02, wspace=0.02)
fig.savefig(f'/home/ryali93/Desktop/landslide4sense2022/docs/figures_n/{model_name}.png', bbox_inches='tight', pad_inches=0)
plt.show()

