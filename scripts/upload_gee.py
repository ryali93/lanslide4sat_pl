import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import numpy as np
import scipy.ndimage
from google.cloud import storage
from torch.utils.data import DataLoader
from dataset import DatasetLandslideEval
from model import LandslideModel
import torch
import os

def descargar_blob(bucket_name, source_blob_name, destination_file_name):
    if os.path.exists(destination_file_name):
        print("File {} already exists. Skipping download.".format(destination_file_name))
        return
    else:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))

def resample(to_path, from_path):
    with rasterio.open(to_path) as to_, rasterio.open(from_path) as from_:
        # Revisar si los sistemas de coordenadas son iguales
        if to_.crs != from_.crs:
            transform, width, height = calculate_default_transform(
                to_.crs, from_.crs, to_.width, to_.height, *to_.bounds)
            kwargs = to_.meta.copy()
            kwargs.update({
                'crs': from_.crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            with rasterio.open('/tmp/hasta_resam.tif', 'w', **kwargs) as hasta_resam:
                for i in range(1, to_.count + 1):
                    reproject(
                        source=rasterio.band(to_, i),
                        destination=rasterio.band(hasta_resam, i),
                        src_transform=to_.transform,
                        src_crs=to_.crs,
                        dst_transform=transform,
                        dst_crs=from_.crs,
                        resampling=Resampling.nearest)
            hasta_resam = rasterio.open('/tmp/hasta_resam.tif', 'r')
        else:
            hasta_resam = to_

        # Checar si las extensiones son iguales
        if hasta_resam.bounds != from_.bounds:
            data = hasta_resam.read(
                out_shape=(hasta_resam.count, from_.height, from_.width),
                resampling=Resampling.nearest)
            transform = from_.transform
        else:
            data = hasta_resam.read()
            transform = hasta_resam.transform

        kwargs = from_.meta.copy()
        kwargs.update({
            'height': from_.height,
            'width': from_.width,
            'transform': transform
        })
        return data


# def split_in_patches(image_data, size=128):
#     n_bandas, width, height = image_data.shape
#     n_parches_x = width // size
#     n_parches_y = height // size
#     parches = []
#     for i in range(n_parches_x):
#         for j in range(n_parches_y):
#             parche = image_data[:, i*size:(i+1)*size, j*size:(j+1)*size]
#             parche_rellenado = rellenar_nan(parche)
#             parches.append(parche_rellenado)
#     return parches

def split_in_patches(image_data, size=128, padding_value=np.nan):
    n_bandas, width, height = image_data.shape
    
    # Calcular el número de parches necesarios en cada dimensión
    n_parches_x = -(-width // size)  # Equivalente a ceiling division
    n_parches_y = -(-height // size)
    
    parches = []
    for i in range(n_parches_x):
        for j in range(n_parches_y):
            x_start, x_end = i*size, (i+1)*size
            y_start, y_end = j*size, (j+1)*size
            
            # Comprobar si estamos en un parche al final que pueda ser más pequeño que size
            if x_end > width or y_end > height:
                parche = np.full((n_bandas, size, size), padding_value, dtype=image_data.dtype)
                x_slice = slice(x_start, min(x_end, width))
                y_slice = slice(y_start, min(y_end, height))
                parche[:, :min(x_end, width)-x_start, :min(y_end, height)-y_start] = image_data[:, x_slice, y_slice]
            else:
                parche = image_data[:, x_start:x_end, y_start:y_end]
            
            parche_rellenado = rellenar_nan(parche)
            parches.append(parche_rellenado)
    return parches



def rellenar_nan(parche):
    for i in range(parche.shape[0]):
        banda = parche[i, :, :]
        mask_nan = np.isnan(banda)
        distancias, indices = scipy.ndimage.distance_transform_edt(mask_nan, return_indices=True)
        banda_copia = banda.copy()
        banda[mask_nan] = banda_copia[indices[0, mask_nan], indices[1, mask_nan]]
        parche[i, :, :] = banda
    return parche

def get_model_pl(model_path):
    model = LandslideModel()
    return model.load_from_checkpoint(checkpoint_path=model_path)

# def reconstruir_mascara(parches, size, original_shape):
#     # _, height, width = original_shape
#     _, width, height = original_shape
#     mascara_reconstruida = np.zeros(original_shape, dtype=parches[0].dtype)
#     n_parches_x = -(-width // size)
#     n_parches_y = -(-height // size)
#     contador = 0
#     for i in range(n_parches_x):
#         for j in range(n_parches_y):
#             if contador >= len(parches):
#                 print("Advertencia: No hay suficientes parches para llenar la imagen. Se rellenará con ceros.")
#                 break
#             parche = parches[contador]
#             if parche.ndim == 3: parche = parche[0]
#             espacio_restante_x = max(min(size, width - i * size), 0)
#             espacio_restante_y = max(min(size, height - j * size), 0)
#             # parche_ajustado = parche[:espacio_restante_y, :espacio_restante_x]
#             parche_ajustado = parche[:espacio_restante_x, :espacio_restante_y]
#             if parche_ajustado.shape != (espacio_restante_x, espacio_restante_y):#(espacio_restante_y, espacio_restante_x):
#                 # parche_ajustado = np.pad(parche_ajustado, ((0, max(espacio_restante_y - parche_ajustado.shape[0], 0)), (0, max(espacio_restante_x - parche_ajustado.shape[1], 0))), 'constant')
#                 parche_ajustado = np.pad(parche_ajustado, ((0, max(espacio_restante_x - parche_ajustado.shape[1], 0)), (0, max(espacio_restante_y - parche_ajustado.shape[0], 0))), 'constant')
#             # mascara_reconstruida[0, j * size:j * size + espacio_restante_y, i * size:i * size + espacio_restante_x] = parche_ajustado
#             mascara_reconstruida[0, i * size:i * size + espacio_restante_x, j * size:j * size + espacio_restante_y] = parche_ajustado
#             contador += 1
#     return mascara_reconstruida
def reconstruir_mascara(parches, size, original_shape):
    _, height, width = original_shape
    mascara_reconstruida = np.zeros(original_shape, dtype=parches[0].dtype)
    n_parches_x = -(-width // size)
    n_parches_y = -(-height // size)
    contador = 0
    for j in range(n_parches_x):
        for i in range(n_parches_y):
            parche = parches[contador] if contador < len(parches) else np.zeros((size, size), dtype=parches[0].dtype)
            if parche.ndim == 3:
                parche = parche[0]
            espacio_restante_x = max(min(size, width - i * size), 0)
            espacio_restante_y = max(min(size, height - j * size), 0)
            parche_ajustado = parche[:espacio_restante_y, :espacio_restante_x]
            if parche_ajustado.shape != (espacio_restante_y, espacio_restante_x):
                parche_ajustado = np.pad(parche_ajustado, ((0, max(espacio_restante_y - parche_ajustado.shape[0], 0)), (0, max(espacio_restante_x - parche_ajustado.shape[1], 0))), 'constant')
            mascara_reconstruida[0, j * size:j * size + espacio_restante_y, i * size:i * size + espacio_restante_x] = parche_ajustado
            contador += 1
    if contador < len(parches):
        print("Advertencia: No hay suficientes parches para llenar la imagen. Se rellenará con ceros.")
    return mascara_reconstruida

def stitch_patches_into_image(patches, original_shape, patch_size=128):
    """
    Une los parches de predicción para reconstruir la imagen original.

    Parameters:
    - patches (list of np.array): Lista de parches de predicción con dimensiones (C, H, W).
    - original_shape (tuple): Dimensión original de la imagen (C, H, W).
    - patch_size (int, optional): Tamaño de los parches cuadrados. Por defecto es 128.

    Returns:
    - np.array: Imagen reconstruida.
    """
    # Crear una matriz vacía con las dimensiones de la imagen original
    stitched_image = np.zeros(original_shape, dtype=np.float32)

    # Número de parches en las dimensiones x e y
    patches_x = -(-original_shape[2] // patch_size) # Estas fórmulas son equivalentes al "ceiling division"
    patches_y = -(-original_shape[1] // patch_size)

    # Número esperado de parches
    expected_patches = patches_x * patches_y
    assert len(patches) == expected_patches, f"Expected {expected_patches} patches, but got {len(patches)}."
    
    patch_index = 0
    
    for i in range(patches_y):
        for j in range(patches_x):
            start_x = j * patch_size
            start_y = i * patch_size
            
            end_x = min((j + 1) * patch_size, original_shape[2])
            end_y = min((i + 1) * patch_size, original_shape[1])
            
            # Determinar el tamaño real del parche (puede ser menor que patch_size si está en el borde)
            real_patch_width = end_x - start_x
            real_patch_height = end_y - start_y

            # Imprimir información de diagnóstico
            print(f"Processing patch {patch_index} at coordinates ({start_x}, {start_y}) to ({end_x}, {end_y}).")
            
            # Extraer el parche real de la lista de parches
            real_patch = patches[patch_index][:, :real_patch_height, :real_patch_width]
            
            # Asignar el parche a la posición correspondiente en la imagen original
            stitched_image[:, start_y:end_y, start_x:end_x] = real_patch

            patch_index += 1
    
    return stitched_image

def guardar_prediccion_como_geotiff(input_path, output_path, prediction):
    with rasterio.open(input_path) as src:
        meta = src.meta
    meta.update(count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction.astype(rasterio.float32), 1)
    print("Prediction saved to {}.".format(output_path))

def main():
    # Bucket config
    bucket_name = 'rgee_dev' # GCS
    set_dir = "test_upload8/" # Local
    s2_source_blob_name = 'tesis/ld_s2_6b_2019_aoi8.tif'
    ap_source_blob_name = 'tesis/ld_ap_aoi8.tif'
    s2_path = set_dir + 'ld_s2_6b_2019_aoi8.tif'
    ap_path = set_dir + 'ld_ap_aoi8.tif'
    output_path = 'test_upload8/pred_vanilla_6b_2019_aoi8.tif'

    # model_path = 'models_n/unet_vanilla_6b_l4spe_nn_1.ckpt'
    # model_path = 'models_n/unet_resnet34_6b_l4spe_nn.ckpt'
    # model_path = 'models/unet_segformer1_6b_full.ckpt'
    # model_path = 'models/unet_resnet34_14b_full_2.ckpt'
    # model_path = 'models/unet_mobilenetv2_6b_full.ckpt'
    model_path = 'models/unet_vanilla_6b_full.ckpt'

    descargar_blob(bucket_name, s2_source_blob_name, s2_path)
    descargar_blob(bucket_name, ap_source_blob_name, ap_path)

    d_s2 = rasterio.open(s2_path).read()
    ap_resamp = resample(ap_path, s2_path)
    ap_resamp = np.moveaxis(ap_resamp, 0, -1)
    ap_resamp = ap_resamp[:, :, :2]
    ap_resamp = np.moveaxis(ap_resamp, -1, 0)
    data = np.concatenate((d_s2, ap_resamp), axis=0)

    parches = split_in_patches(data)
    model = get_model_pl(model_path)
    dataset_eval = DatasetLandslideEval(parches)
    loader = DataLoader(dataset_eval, batch_size=16)

    all_predictions = []
    model.eval()
    with torch.no_grad():
        for images in loader:
            images = images.to(model.device)
            predictions = model(images)
            predictions = torch.sigmoid(predictions)
            predictions = predictions.cpu().numpy()
            all_predictions.append(predictions)

    parches_aplanados = [parche for batch in all_predictions for parche in batch]
    reconstructed_image = stitch_patches_into_image(parches_aplanados, data.shape)
    guardar_prediccion_como_geotiff(s2_path, output_path, reconstructed_image[0])


if __name__ == "__main__":
    main()
