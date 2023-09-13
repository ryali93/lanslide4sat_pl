import time
import ee
import geemap
import tensorflow as tf
import torch
from typing import List
import torchdata.datapipes.iter as dp
from google.cloud import storage

ee.Initialize()

norm_dicc = {
    "own": {
        "B1": [681.0, 3165.0],
        "B2": [183.0, 8655.0],
        "B3": [258.0, 7697.0],
        "B4": [68.0, 6779.0],
        "B5": [1.0, 5293.0],
        "B6": [0.0, 5600.0],
        "B7": [58.0, 6544.0],
        "B8": [111.0, 8930.0],
        "B9": [36.0, 2972.0],
        "B10": [1.0, 270.0],
        "B11": [36.0, 7280.0],
        "B12": [1.0, 12937.0],
        "B13": [0.0, 1.4837092161178589],
        "B14": [0.000001, 4128.0]
    }
}

def list_blobs(bucket_name, prefix=None):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    return [blob.name for blob in blobs]

def normalize_minmax(data, band, norm_min, norm_max):
    band_img = data.expression(
        f'(b("{band}") - {norm_min})/({norm_max} - {norm_min})', {
            band: data.select(band)
        }).rename(band).toFloat().select(band)
    return band_img

def get_imgs(extent):
    # extent = [-76.27886, -10.522948, -76.07424, -10.357505]
    xmin,ymin,xmax,ymax = extent
    pol = ee.Geometry.Rectangle([xmin,ymin,xmax,ymax])

    uriBase = 'gs://rgee_dev/COG/'
    collection = ee.ImageCollection(ee.List([
        ee.Image.loadGeoTIFF(uriBase + 'AP_26505_FBS_F6970_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26505_FBS_F7000_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26505_FBS_F6960_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26505_FBS_F6990_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26505_FBS_F6980_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F7000_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F6990_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26505_FBS_F6950_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F7010_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F7020_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F6980_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_24988_FBD_F7010_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F6970_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26082_FBS_F6970_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26257_FBS_F6970_RT1.cog.tif'),
        ee.Image.loadGeoTIFF(uriBase + 'AP_26257_FBS_F6960_RT1.cog.tif')
    ]))

    dem = collection.filterBounds(pol).first().clip(pol).select("B0").clip(pol).toFloat().rename("B14")
    slope = ee.Terrain.slope(dem).select("slope").rename("B13")

    d_s2 = ee.ImageCollection('COPERNICUS/S2_SR')\
                .filterDate('2020-06-01', '2020-07-30')\
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))\
                .filterBounds(pol)\
                .first()\
                .select("B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")\
                .clip(pol)
    ndvi = d_s2.normalizedDifference(['B8','B4']).rename("B15")
    dataset = d_s2.addBands(slope).addBands(dem).addBands(ndvi)

    band2 = normalize_minmax(dataset, "B2", norm_dicc["own"]["B2"][0], norm_dicc["own"]["B2"][1])
    band3 = normalize_minmax(dataset, "B3", norm_dicc["own"]["B3"][0], norm_dicc["own"]["B3"][1])
    band4 = normalize_minmax(dataset, "B4", norm_dicc["own"]["B4"][0], norm_dicc["own"]["B4"][1])
    band13 = normalize_minmax(dataset, "B13", norm_dicc["own"]["B13"][0], norm_dicc["own"]["B13"][1])
    band14 = normalize_minmax(dataset, "B14", norm_dicc["own"]["B14"][0], norm_dicc["own"]["B14"][1])

    dataset_img = band2.addBands(band3).addBands(band4).addBands(band13).addBands(band14).addBands(ndvi)
    return dataset_img

def download_imgs_to_gcs(data, area, bucket_name, prefix, monitor=False):
    # outputBucket = 'rgee_dev'
    # imageFilePrefix = 'tesis/ld'
    outputBucket = bucket_name
    imageFilePrefix = prefix
    # Specify patch and file dimensions.
    imageExportFormatOptions = {
    'patchDimensions': [128, 128],
    'compressed': True
    }
    # Setup the task.
    imageTask = ee.batch.Export.image.toCloudStorage(
    image=data,
    description='Image Export',
    fileNamePrefix=imageFilePrefix,
    bucket=outputBucket,
    scale=10,
    fileFormat='TFRecord',
    region=area.getInfo()['coordinates'],
    formatOptions=imageExportFormatOptions,
    )
    imageTask.start()

    if monitor:
        while imageTask.active():
            print('Polling for task (id: {}).'.format(imageTask.id))
            time.sleep(5)


# def predict_input_fn(fileNames, side, bands):

#   # Read `TFRecordDatasets`
#   dataset = tf.data.TFRecordDataset(fileNames, compression_type='GZIP')
#   featuresDict = {x:tf.io.FixedLenFeature([side, side], dtype=tf.float32) for x in bands}

#   # Make a parsing function
#   def parse_image(example_proto):
#     parsed_features = tf.io.parse_single_example(example_proto, featuresDict)
#     return parsed_features

#   def stack_images(features):
#     nfeat = tf.transpose(tf.squeeze(tf.stack(list(features.values()))))
#     return nfeat

#   dataset = dataset.map(parse_image, num_parallel_calls=4)
#   dataset = dataset.map(stack_images, num_parallel_calls=4)
#   dataset = dataset.batch(side*side)
#   return dataset


def predict_input_fn(fileNames: List[str], side: int, bands: List[str]) -> dp.IterDataPipe:
    # Define la especificación para decodificar el TFRecord
    spec = { band: (torch.float32, (side, side)) for band in bands } 
    # Crea un DataPipe que emite rutas de archivo
    files_dp = dp.FileLister(fileNames)
    # Carga y decodifica los TFRecords
    tfrecord_dp = dp.TFRecordLoader(files_dp, spec=spec)
    return tfrecord_dp

# def predict_input_fn(fileNames: List[str], side: int, bands: List[str]) -> dp.IterDataPipe:
#     # Define la especificación para decodificar el TFRecord
#     spec = { band: (torch.float32, (side, side)) for band in bands } 
#     # Aquí, en lugar de usar la ruta de GCS, utiliza el local_path
#     files_dp = dp.FileLister([local_path])  # Asegúrate de que sea una lista
#     # Carga y decodifica los TFRecords
#     tfrecord_dp = dp.TFRecordLoader(files_dp, spec=spec)
#     return tfrecord_dp


def ingest_gee(outputAssetId, gcs_image_uri, mixer):
    # outputAssetId = 'your_output_asset_id'
    # gcs_image_uri = 'gs://your_bucket/your_image.tif'
    asset_request = {
    'id': outputAssetId,
    'tilesets': [
        {
        'sources': [{'primaryPath': gcs_image_uri}]
        }
    ],
    'crs': mixer["projection"]["crs"],
    'affine': mixer["projection"]["affine"]["doubleMatrix"],
    }
    ee.data.startIngestion(ee.data.newTaskId(), asset_request)