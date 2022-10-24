Resamplear <- function(hasta, desde){
  if(proj4string(desde) != proj4string(hasta)){
    hasta_proj = projectRaster(hasta, crs=CRS(proj4string(desde)))
  }else{
    hasta_proj = hasta
  }
  if(extent(desde) != extent(hasta_proj)){
    hasta_resam = raster::resample(hasta_proj, desde, method="ngb")
  }else{
    hasta_resam = hasta_proj
  }
  return(hasta_resam)
}

radians <- function(img){
  return(img$toFloat()$multiply(pi)$divide(180))
}

read_point <- function(path){
  # Read point path
  point = st_read(path)
  point = st_zm(point, drop = TRUE, what = "ZM")
  point_utm = st_transform(point, crs = 32718)
}

generate_patch <- function(point, w, h){
  # Create polygon from coordinates
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
  # Download Sentinel2 bands
  dataset = ee$ImageCollection('COPERNICUS/S2_SR')$
    filterDate('2018-01-01', '2020-01-30')$
    filter(ee$Filter$lt('CLOUDY_PIXEL_PERCENTAGE', 10))$
    filterBounds(pol)$
    first()$
    select("B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")
  
  patch_image = ee_as_raster(image=dataset, region=pol, via = "drive", scale = res, quiet=T)
  return(patch_image)
}

download_srtm <- function(pol, res=10){
  # Download srtm
  dataset = ee$Image('USGS/SRTMGL1_003')$
    select("elevation")$
    clip(pol)$toFloat()
  
  slope = radians(ee$Terrain$slope(dataset))$toFloat()
  dataset_n = dataset$addBands(slope)
  
  patch_image = ee_as_raster(image=dataset_n, region=pol, via = "drive", scale = res, crs="EPSG:32718", quiet=T)
  return(patch_image)
}

create_dir_general <- function(){
  # 1. Create principal folder
  dir.create(
    path = "landslide/landslide.iris/segmentation",
    showWarnings = FALSE, 
    recursive = TRUE
  )
}


create_dir_point <- function(path_point){
  # 1. Create dir for each point
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
  # Create pol for download
  pol_sf = st_sfc(polygon, crs=32718)
  pol_wgs = st_transform(pol_sf, 4326)
  pol_ee = sf_as_ee(pol_wgs)
}

merge_img <- function(sen2, dem){
  b1_b14 = addLayer(sen2, dem)
  return(brick(b1_b14))
}

export_h5 <- function(ras){
  # Export img to h5 file
  nx = minmax(ras)
  rn = (ras - nx[1,]) / (nx[2,] - nx[1,])
  ar = as.array(rn)
}

create_metadata <- function(point, scene_id){
  coords = st_coordinates(point)
  lj = list("spacecraft_id" = "Sentinel2/SRTM",
            "scene_id" = scene_id,
            "location" = c(coords[2], coords[1]),
            "resolution" = 10
  )
  j = toJSON(lj, pretty = T, auto_unbox = T)
}
