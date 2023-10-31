import ee
import time
import rasterio
import numpy as np
from google.cloud import storage
import scipy.ndimage

ee.Initialize()

def define_aoi(xmin, ymin, xmax, ymax):
    return ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

def load_alos_palsar_images(uri_base, image_list):
    collection = ee.ImageCollection(ee.List([
        ee.Image.loadGeoTIFF(uri_base + image_name) for image_name in image_list
    ]))
    return collection

def add_cloud_bands(img, cld_prb_thresh):
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
    is_cloud = cld_prb.gt(cld_prb_thresh).rename('clouds')
    return img.addBands(ee.Image([cld_prb, is_cloud]))

def add_shadow_bands(img):
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
    not_cld_shdw = img.select('cloudmask').Not()
    # return img.select('B.*').updateMask(not_cld_shdw)
    # return img.updateMask(not_cld_shdw).select(['B[2-8]', None, False])
    return img.updateMask(not_cld_shdw).select(['B2','B3','B4','B8']);


def get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter):
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
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
    s2_sr_cld_col = get_s2_sr_cld_col(aoi, start_date, end_date, cloud_filter)
    s2_sr_median = (s2_sr_cld_col.map(add_cld_shdw_mask)
                                .map(apply_cld_shdw_mask)
                                .median())
    d_s2 = s2_sr_median.clip(aoi).select(bands).toFloat()
    # d_s2 = s2_sr_median.select("B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12").clip(aoi).toFloat()
    return d_s2

def export_to_cloud_storage(image, output_bucket, file_prefix, aoi, scale):
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
    while task.active():
        print('Polling for task (id: {}).'.format(task.id))
        time.sleep(10)

def download_image_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def main():
    start_date = '2022-09-01'
    end_date = '2022-10-30'
    # xmin, ymin, xmax, ymax = [-75.583,-10.813,-75.317,-10.637]
    # xmin, ymin, xmax, ymax = [-75.621,-10.520,-75.400,-10.338]
    # xmin, ymin, xmax, ymax = [-75.684,-10.457,-75.466,-10.272]
    xmin, ymin, xmax, ymax = [-75.359,-10.879,-75.221,-10.790]
    bands = ['B2','B3','B4','B8']
    # xmin, ymin, xmax, ymax = [-75.249597, -10.770447, -74.966013, -10.503883]
    aoi = define_aoi(xmin, ymin, xmax, ymax)
    cloud_filter = 30
    dataset_s2 = process_sentinel2_images(aoi, start_date, end_date, bands, cloud_filter)
    dataset_ap = process_alos_palsar_images(aoi)

    output_bucket = 'rgee_dev'
    
    file_prefix_s2 = 'tesis/ld_s2_6b_2022_aoi6'
    file_prefix_ap = 'tesis/ld_ap_aoi6'
    scale_s2 = 10
    scale_ap = 12.5

    task = export_to_cloud_storage(dataset_s2, output_bucket, file_prefix_s2, aoi, scale_s2)
    monitor_task(task)
    # task = export_to_cloud_storage(dataset_ap, output_bucket, file_prefix_ap, aoi, scale_ap)
    # monitor_task(task)  

    # download_image_from_gcs('rgee_dev', 'tesis4/ld_2019.tif', '/tmp/ld_2019.tif')
    # download_image_from_gcs('rgee_dev', 'tesis4/ld_2019_ap.tif', '/tmp/ld_2019_ap.tif')

if __name__ == "__main__":
    main()