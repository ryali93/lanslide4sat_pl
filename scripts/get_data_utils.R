ResampleRaster <- function(target, source){
  #' ResampleRaster function
  #' Resamples a raster to match the extent and resolution of another raster
  #' @param target The raster to be resampled
  #' @param source The raster to be used as a template for the resampling
  #' @return The resampled raster
  #' @export

  if(proj4string(source) != proj4string(target)){
    target_proj = projectRaster(target, crs=CRS(proj4string(source)))
  }else{
    target_proj = target
  }
  if(extent(source) != extent(target_proj)){
    target_resampled = raster::resample(target_proj, source, method="ngb")
  }else{
    target_resampled = target_proj
  }
  return(target_resampled)
}

radians <- function(img){
  #' Convert degrees to radians
  #' @param img An image object
  #' @return The input image converted to radians

  return(img$toFloat()$multiply(pi)$divide(180))
}

read_point <- function(path){
  #' Reads a point shapefile and transforms it to UTM projection
  #' @param path A character string indicating the path to the shapefile
  #' @return A spatial object in UTM projection

  point = st_read(path)
  point = st_zm(point, drop = TRUE, what = "ZM")
  point_utm = st_transform(point, crs = 32718)
}

generate_patch <- function(point, w, h){
  #' Generates a polygon of width w and height h centered on a point
  #' @param point A spatial object
  #' @param w A numeric value indicating the width of the polygon
  #' @param h A numeric value indicating the height of the polygon
  #' @return A polygon object

  x = st_coordinates(point)[1]
  y = st_coordinates(point)[2]
  w_m = w/2
  h_m = h/2
  
  pol = st_polygon(
    list(cbind(
      c(x - w_m, x + w_m, x + w_m, x - w_m, x - w_m),
      c(y + h_m, y + h_m, y - h_m, y - h_m, y + h_m)))
  )
  return(pol)
}

download_sen2 <- function(pol, res=10){
  #' Downloads Sentinel-2 images from Google Earth Engine
  #' @param pol A polygon object
  #' @param res A numeric value indicating the resolution of the image
  #' @return A raster object

  dataset = ee$ImageCollection('COPERNICUS/S2')$
    filterDate('2018-01-01', '2020-01-30')$
    filter(ee$Filter$lt('CLOUDY_PIXEL_PERCENTAGE', 10))$
    filterBounds(pol)$
    first()$
    select("B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12")
  
  patch_image = ee_as_raster(image=dataset, region=pol, via = "drive", scale = res, quiet=T)
  return(patch_image)
}

download_srtm <- function(pol, res=10){
  #' Downloads SRTM images from Google Earth Engine
  #' @param pol A polygon object
  #' @param res A numeric value indicating the resolution of the image
  #' @return A raster object

  dataset = ee$Image('USGS/SRTMGL1_003')$
    select("elevation")$
    clip(pol)$toFloat()
  
  slope = radians(ee$Terrain$slope(dataset))$toFloat()
  dataset_n = dataset$addBands(slope)
  
  patch_image = ee_as_raster(image=dataset_n, region=pol, via = "drive", scale = res, crs="EPSG:32718", quiet=T)
  return(patch_image)
}

download_alos <- function(pol, res=10){
  #' Downloads ALOS images from Google Earth Engine
  #' @param pol A polygon object
  #' @param res A numeric value indicating the resolution of the image
  #' @return A raster object
  
  uriBase = 'gs://rgee_dev/COG/'
  collection = ee$ImageCollection(list.files(path = uriBase, pattern = "^AP.*tif$", full.names = TRUE) %>% 
                                    purrr::map(ee$Image$loadGeoTIFF) %>% 
                                    ee$ImageCollection)
  collectionMosaik = collection$mosaic()
  dem = collectionMosaik$
    clip(pol)$
    select("B0")$
    toFloat()
  
  patch_image = ee_as_raster(image=dem, region=pol, via = "drive", scale = res, crs="EPSG:32718", quiet=T)
  return(patch_image)
}

create_dir_general <- function(){
  #' Creates the general directory structure
  #' @return NULL
  dir.create(
    path = "landslide/landslide.iris/segmentation",
    showWarnings = FALSE, 
    recursive = TRUE
  )
}

create_dir_point <- function(path_point){
  #' Creates the directory structure for a point
  #' @param path_point A character string indicating the path to the point

  dir.create(
    path = sprintf("landslide/%s/input", path_point),
    showWarnings = FALSE,
    recursive = TRUE
  )
  
  dir.create(
    path = sprintf("landslide/%s/target", path_point),
    showWarnings = FALSE,
    recursive = TRUE
  )
}

create_pol <- function(pol){
  #' Creates a polygon object in Google Earth Engine
  #' @param pol A polygon object
  #' @return A polygon object in Google Earth Engine

  pol_sf = st_sfc(polygon, crs=32718)
  pol_wgs = st_transform(pol_sf, 4326)
  pol_ee = sf_as_ee(pol_wgs)
}

merge_img <- function(sen2, dem){
  #' Merges Sentinel-2 and SRTM images
  #' @param sen2 A raster object
  #' @param dem A raster object
  #' @return A raster object

  b1_b14 = addLayer(sen2, dem)
  return(brick(b1_b14))
}

export_h5 <- function(ras){
  #' Exports a raster object to a h5 file
  #' @param ras A raster object
  #' @return A h5 file

  nx = minmax(ras)
  rn = (ras - nx[1,]) / (nx[2,] - nx[1,])
  ar = as.array(rn)
}

create_metadata <- function(point, scene_id){
  #' Creates a metadata file
  #' @param point A spatial object
  #' @param scene_id A character string indicating the scene id
  #' @return A metadata file
  
  coords = st_coordinates(point)
  lj = list("spacecraft_id" = "Sentinel2/SRTM",
            "scene_id" = scene_id,
            "location" = c(coords[2], coords[1]),
            "resolution" = 10
  )
  j = toJSON(lj, pretty = T, auto_unbox = T)
}
