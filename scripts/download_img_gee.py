import ee
import time
import rasterio
import numpy as np
from google.cloud import storage
import scipy.ndimage

ee.Initialize()

def load_alos_palsar_images(uri_base, image_list):
    """
    Loads ALOS PALSAR images from a given URI base and list of image names.

    Args:
        uri_base (str): The base URI where the images are located.
        image_list (list): A list of image names to load.

    Returns:
        ee.ImageCollection: A collection of loaded images.
    """
    collection = ee.ImageCollection(ee.List([
        ee.Image.loadGeoTIFF(uri_base + image_name) for image_name in image_list
    ]))
    return collection

def add_cloud_bands(img, cld_prb_thresh):
    """
    Adds cloud probability and binary cloud mask bands to an input image.

    Args:
        img: An input image.
        cld_prb_thresh: Cloud probability threshold value.

    Returns:
        An image with added cloud probability and binary cloud mask bands.
    """
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(cld_prb_thresh).rename('clouds')
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
    """
    Adds shadow bands to the input image.

    Args:
    img: ee.Image
        The input image.

    Returns:
    ee.Image
        The input image with shadow bands added.
    """
    not_water = img.select('SCL').neq(6)
    SR_BAND_SCALE = 1e4
    NIR_DRK_THRESH = 0.15
    CLD_PRJ_DIST = 1
    dark_pixels = img.select('B8').lt(NIR_DRK_THRESH*SR_BAND_SCALE).multiply(not_water).rename('dark_pixels')
    shadow_azimuth = ee.Number(90).subtract(ee.Number(img.get('MEAN_SOLAR_AZIMUTH_ANGLE')));
    cld_proj = (img.select('clouds').directionalDistanceTransform(shadow_azimuth, CLD_PRJ_DIST*10)
        .reproject(**{'crs': img.select(0).projection(), 'scale': 100})
        .select('distance')
        .mask()
        .rename('cloud_transform'))
    shadows = cld_proj.multiply(dark_pixels).rename('shadows')
    return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

def add_cld_shdw_mask(img):
    """
    Adds a cloud and shadow mask to the input image.

    Args:
        img: An ee.Image object representing the input image.

    Returns:
        An ee.Image object with cloud and shadow masks added.
    """
    BUFFER = 50
    CLD_PRB_THRESH = 50
    img_cloud = add_cloud_bands(img, CLD_PRB_THRESH)
    img_cloud_shadow = add_shadow_bands(img_cloud)
    is_cld_shdw = img_cloud_shadow.select('clouds').add(img_cloud_shadow.select('shadows')).gt(0)
    is_cld_shdw = (is_cld_shdw.focalMin(2).focalMax(BUFFER*2/20)
        .reproject(**{'crs': img.select([0]).projection(), 'scale': 20})
        .rename('cloudmask'))
    return img_cloud_shadow.addBands(is_cld_shdw)

def apply_cld_shdw_mask(img):
    """
    Applies a cloud and shadow mask to the input image.

    Args:
        img: An ee.Image object.

    Returns:
        An ee.Image object with the cloud and shadow mask applied.
    """
    not_cld_shdw = img.select('cloudmask').Not()
    return img.updateMask(not_cld_shdw).select(['B2','B3','B4','B8']);

def get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter):
    """
    Returns an ImageCollection of Sentinel-2 Surface Reflectance data with cloud probability band.

    Args:
        aoi (ee.Geometry): Area of interest.
        start_date (str): Start date of the time range in yyyy-mm-dd format.
        end_date (str): End date of the time range in yyyy-mm-dd format.
        cloud_filter (int): Maximum cloud cover percentage allowed.

    Returns:
        ee.ImageCollection: ImageCollection of Sentinel-2 Surface Reflectance data with cloud probability band.
    """
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', cloud_filter)))
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def process_alos_palsar_images(aoi):
    """
    This function processes ALOS PALSAR images by loading them from Google Cloud Storage, 
    creating an image collection, mosaicking the images, clipping the mosaic to the area of interest, 
    and calculating the slope and DEM. The resulting dataset is returned.
    
    Args:
    aoi: ee.Geometry.Polygon - Area of interest polygon
    
    Returns:
    dataset_ap: ee.Image - Processed ALOS PALSAR dataset
    """
    uri_base = 'gs://rgee_dev/COG/'
    alos_palsar_images = [
        'AP_26505_FBS_F6970_RT1.cog.tif',
        'AP_26505_FBS_F7000_RT1.cog.tif',
        'AP_26505_FBS_F6960_RT1.cog.tif',
        'AP_26505_FBS_F6990_RT1.cog.tif',
        'AP_26505_FBS_F6980_RT1.cog.tif',
        'AP_26082_FBS_F7000_RT1.cog.tif',
        'AP_26082_FBS_F6990_RT1.cog.tif',
        'AP_26505_FBS_F6950_RT1.cog.tif',
        'AP_26082_FBS_F7010_RT1.cog.tif',
        'AP_26082_FBS_F7020_RT1.cog.tif',
        'AP_26082_FBS_F6980_RT1.cog.tif',
        'AP_24988_FBD_F7010_RT1.cog.tif',
        'AP_26082_FBS_F6970_RT1.cog.tif',
        'AP_26082_FBS_F6970_RT1.cog.tif',
        'AP_26257_FBS_F6970_RT1.cog.tif',
        'AP_26257_FBS_F6960_RT1.cog.tif'
    ]
    collection = ee.ImageCollection(ee.List([
        ee.Image.loadGeoTIFF(uri_base + image_name) for image_name in alos_palsar_images
    ]))
    mosaic = collection.mosaic()
    dem = mosaic.clip(aoi).select("B0").toFloat().rename("B14").reproject('EPSG:4326', None, 12.5)
    slope = ee.Terrain.slope(dem).select("slope").toFloat().rename("B13")
    dataset_ap = slope.addBands(dem)
    return dataset_ap

def process_sentinel2_images(aoi, start_date, end_date, bands, cloud_filter):
    """
    Processes Sentinel-2 images for a given area of interest (aoi) and time range (start_date to end_date).
    Applies a cloud filter to the images and selects the specified bands. Returns the processed images as a float.
    
    Args:
    - aoi: ee.Geometry - area of interest
    - start_date: str - start date of time range in yyyy-mm-dd format
    - end_date: str - end date of time range in yyyy-mm-dd format
    - bands: list - list of band names to select
    - cloud_filter: float - maximum cloud cover percentage
    
    Returns:
    - d_s2: ee.Image - processed Sentinel-2 images as a float
    """
    s2_sr_cld_col = get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter)
    s2_sr_median = s2_sr_cld_col.first()
    d_s2 = s2_sr_median.clip(aoi).select(bands).toFloat()
    return d_s2

def export_to_cloud_storage(image, output_bucket, file_prefix, aoi, scale):
    """
    Exports an Earth Engine image to Google Cloud Storage.

    Args:
        image (ee.Image): The Earth Engine image to export.
        output_bucket (str): The name of the Google Cloud Storage bucket to export to.
        file_prefix (str): The prefix to use for the output file name.
        aoi (ee.Geometry): The area of interest to export.
        scale (int): The scale of the exported image in meters.

    Returns:
        ee.batch.Task: The Earth Engine task representing the export process.
    """
    task = ee.batch.Export.image.toCloudStorage(
      image=image,
      description='Image Export',
      fileNamePrefix=file_prefix,
      bucket=output_bucket,
      scale=scale,
      fileFormat='GeoTIFF',
      region=aoi.getInfo()['coordinates'],
      maxPixels=2e8
    )
    task.start()
    return task

def monitor_task(task):
    """
    Monitors the progress of a task until it is completed.

    Args:
        task: An instance of a task object.

    Returns:
        None
    """
    while task.active():
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(10)

def download_image_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """
    Downloads an image from Google Cloud Storage to a local file.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob to download.
        destination_file_name (str): The local file path to save the downloaded image.

    Returns:
        None
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def main():
    start_date = '2019-06-01'
    end_date = '2019-06-30'
    xmin, ymin, xmax, ymax = [-75.390,-11.292,-75.344,-11.250] # aoi example
    bands = ['B2','B3','B4','B8']
    aoi = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])
    cloud_filter = 100
    dataset_s2 = process_sentinel2_images(aoi, start_date, end_date, bands, cloud_filter)
    dataset_ap = process_alos_palsar_images(aoi)

    output_bucket = 'rgee_dev'
    
    file_prefix_s2 = 'l4spe/ld_s2_6b_2019_aoi8'
    file_prefix_ap = 'l4spe/ld_ap_aoi8'
    scale_s2 = 10
    scale_ap = 12.5

    task = export_to_cloud_storage(dataset_s2, output_bucket, file_prefix_s2, aoi, scale_s2)
    monitor_task(task)
    task = export_to_cloud_storage(dataset_ap, output_bucket, file_prefix_ap, aoi, scale_ap)
    monitor_task(task)  

if __name__ == "__main__":
    main()