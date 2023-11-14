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
    """
    Downloads a blob from the given bucket and saves it to the specified file path.

    Args:
        bucket_name (str): The name of the bucket containing the blob to download.
        source_blob_name (str): The name of the blob to download.
        destination_file_name (str): The file path to save the downloaded blob to.

    Returns:
        None
    """
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
    """
    Resamples a raster image from one CRS and resolution to another CRS and resolution.

    Args:
        to_path (str): Filepath of the raster image to be resampled.
        from_path (str): Filepath of the raster image to be used as the reference for resampling.

    Returns:
        numpy.ndarray: The resampled raster image data.

    """
    with rasterio.open(to_path) as to_, rasterio.open(from_path) as from_:
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
            with rasterio.open('/tmp/to_resam.tif', 'w', **kwargs) as to_resam:
                for i in range(1, to_.count + 1):
                    reproject(
                        source=rasterio.band(to_, i),
                        destination=rasterio.band(to_resam, i),
                        src_transform=to_.transform,
                        src_crs=to_.crs,
                        dst_transform=transform,
                        dst_crs=from_.crs,
                        resampling=Resampling.nearest)
            to_resam = rasterio.open('/tmp/to_resam.tif', 'r')
        else:
            to_resam = to_

        if to_resam.bounds != from_.bounds:
            data = to_resam.read(
                out_shape=(to_resam.count, from_.height, from_.width),
                resampling=Resampling.nearest)
            transform = from_.transform
        else:
            data = to_resam.read()
            transform = to_resam.transform

        kwargs = from_.meta.copy()
        kwargs.update({
            'height': from_.height,
            'width': from_.width,
            'transform': transform
        })
        return data

def split_in_patches(image_data, size=128, padding_value=np.nan):
    """
    Divide una imagen en parches cuadrados de tamaño size.

    Parameters:
    - image_data (np.array): Imagen a dividir en parches.
    - size (int, optional): Tamaño de los parches cuadrados. Por defecto es 128.
    - padding_value (int, optional): Valor de relleno para los bordes de los parches. Por defecto es np.nan.

    Returns:
    - list of np.array: Lista de parches de la imagen original.
    """

    n_bandas, width, height = image_data.shape
    
    # Calculate the number of patches needed in each dimension
    n_parches_x = -(-width // size)
    n_parches_y = -(-height // size)
    
    parches = []
    for i in range(n_parches_x):
        for j in range(n_parches_y):
            x_start, x_end = i*size, (i+1)*size
            y_start, y_end = j*size, (j+1)*size
            
            # Check if there are a patch at the end that may be smaller than size
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
    """
    Fills NaN values in a patch with the nearest non-NaN value in the same band.

    Args:
        parche (numpy.ndarray): A 3D array representing a patch of an image.

    Returns:
        numpy.ndarray: The input patch with NaN values filled.
    """
    for i in range(parche.shape[0]):
        banda = parche[i, :, :]
        mask_nan = np.isnan(banda)
        distancias, indices = scipy.ndimage.distance_transform_edt(mask_nan, return_indices=True)
        banda_copia = banda.copy()
        banda[mask_nan] = banda_copia[indices[0, mask_nan], indices[1, mask_nan]]
        parche[i, :, :] = banda
    return parche

def get_model_pl(model_path):
    """
    Loads a PyTorch Lightning model from a checkpoint file.

    Args:
        model_path (str): The path to the checkpoint file.

    Returns:
        A PyTorch Lightning model loaded from the checkpoint file.
    """
    model = LandslideModel()
    return model.load_from_checkpoint(checkpoint_path=model_path)


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
    # Create an empty array with the dimensions of the original image
    stitched_image = np.zeros(original_shape, dtype=np.float32)

    # Number of patches in the x and y dimensions
    patches_x = -(-original_shape[2] // patch_size)
    patches_y = -(-original_shape[1] // patch_size)

    # Number of patches expected
    expected_patches = patches_x * patches_y
    assert len(patches) == expected_patches, f"Expected {expected_patches} patches, but got {len(patches)}."
    
    patch_index = 0
    
    for i in range(patches_y):
        for j in range(patches_x):
            start_x = j * patch_size
            start_y = i * patch_size
            
            end_x = min((j + 1) * patch_size, original_shape[2])
            end_y = min((i + 1) * patch_size, original_shape[1])
            
            # Determinate the real patch size (may be smaller than patch_size if it's on the border)
            real_patch_width = end_x - start_x
            real_patch_height = end_y - start_y

            # Print diagnostic information
            print(f"Processing patch {patch_index} at coordinates ({start_x}, {start_y}) to ({end_x}, {end_y}).")
            
            # Extract the real patch from the list of patches
            real_patch = patches[patch_index][:, :real_patch_height, :real_patch_width]
            
            # Assign the patch to the corresponding position in the original image
            stitched_image[:, start_y:end_y, start_x:end_x] = real_patch

            patch_index += 1
    
    return stitched_image

def save_predict_as_geotiff(input_path, output_path, prediction):
    """
    Saves the prediction as a GeoTIFF file.

    Args:
    input_path (str): The path to the input raster file.
    output_path (str): The path to the output GeoTIFF file.
    prediction (numpy.ndarray): The prediction array.

    Returns:
    None
    """
    with rasterio.open(input_path) as src:
        meta = src.meta
    meta.update(count=1)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(prediction.astype(rasterio.float32), 1)
    print("Prediction saved to {}.".format(output_path))

def main():
    # Bucket config
    bucket_name = 'rgee_dev' # GCS
    set_dir = "test_upload9/" # Local
    s2_source_blob_name = 'l4spe/ld_s2_6b_2019_aoi8.tif'
    ap_source_blob_name = 'l4spe/ld_ap_aoi8.tif'
    s2_path = set_dir + 'ld_s2_6b_2019_aoi8.tif'
    ap_path = set_dir + 'ld_ap_aoi8.tif'
    output_path = 'test_upload9/pred_vanilla_6b_2019_aoi8.tif'

    model_path = 'models/modelo_test.ckpt'

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
    save_predict_as_geotiff(s2_path, output_path, reconstructed_image[0])

if __name__ == "__main__":
    main()
