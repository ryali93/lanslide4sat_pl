library(rgee)
library(tidyverse)
library(sf)
library(raster)
library(terra)
library(jsonlite)

ee_Initialize(drive = T)

setwd("E:/ai/new_dataset")

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

download_sen2 <- function(pol){
  # Download Sentinel2 bands
  dataset = ee$ImageCollection('COPERNICUS/S2_SR')$
    filterDate('2018-01-01', '2020-01-30')$
    filter(ee$Filter$lt('CLOUDY_PIXEL_PERCENTAGE', 10))$
    filterBounds(pol)$
    first()$
    select("B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")
  
  patch_image = ee_as_raster(image=dataset, region=pol, via = "drive", scale = 10)
  return(patch_image)
}

download_alos <- function(pol){
  # Download ALOS PALSAR images
  r1 = raster("AP_25513_PLR_F6960_RT1/AP_25513_PLR_F6960_RT1.dem.tif")
  r1_clip = crop(r1, as(pol, "Spatial"))
  return(r1_clip)
}

download_srtm <- function(pol){
  # Download srtm
  dataset = ee$Image('USGS/SRTMGL1_003')$
    select("elevation")$
    clip(pol)
  
  # slope = ee$Terrain$slope(dataset)$int16()
  # dataset_n = dataset$addBands(slope)

  patch_image = ee_as_raster(image=dataset, region=pol, via = "drive", scale = 10)
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
  dem_res = raster::resample(dem, sen2)
  slope_res = terrain(dem_res, opt="slope")
  
  b1_b14 = addLayer(sen2, dem_res, slope_res)
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
  lj = list("spacecraft_id" = "Sentinel2/AlosPalsar",
       "scene_id" = scene_id,
       "location" = c(coords[2], coords[1]),
       "resolution" = 10
  )
  j = toJSON(lj, pretty = T, auto_unbox = T)
}

# ---------------------------------------------------------------

path_points <- "E:/ai/desliza_f.kml"
# Create folders
create_dir_general()
# Read points
points <- read_point(path_points)

vacios = c()
for(i in rows[4:length(rows)]){ #nrow(points)
  i_path = paste0("point_", str_pad(as.character(i), 3, side = "left", pad = "0"))
  tryCatch({
    if (length(list.files(sprintf("landslide/%s/input", i_path))) >= 14){
      # 1. Create dirs and save points
      point = points[i,]
      point_wgs = st_transform(point, 4326)
      create_dir_point(i_path)
      # st_write(point, dsn = sprintf("landslide/%s/%s.gpkg", i_path, i_path), layer = i_path)
      
      # 2. Create polygons
      polygon = generate_patch(point, 1490, 1490)
      polygon_ee = create_pol(polygon)
      
      # 3. Download satellite images
      # dem_alos = download_alos(polygon)
      dem_srtm = download_srtm(polygon_ee)
      slope_srtm = terrain(dem_srtm[[1]], opt="slope")
      sen2 = raster(sprintf("landslide/%s/input/B13.tif", i_path))
      dem_srtm1 = Resamplear(dem_srtm, sen2)
      slope_srtm1 = Resamplear(slope_srtm, sen2)
      # sen2 = download_sen2(polygon_ee)
      # img = merge_img(sen2, dem_alos)
      
      # img = merge_img(img, dem_srtm)
      # 4. Save imgs
      # for(i in 1:nlayers(img)){
      #   terra::writeRaster(img[[i]], sprintf("landslide/%s/input/B%s.tif", i_path, as.character(i)), overwrite=T)
      # }
      terra::writeRaster(dem_srtm1, sprintf("landslide/%s/input/B15.tif", i_path), overwrite=T)
      terra::writeRaster(slope_srtm1, sprintf("landslide/%s/input/B16.tif", i_path), overwrite=T)
      
      # 5. Export h5 file
      img = brick(stack(sprintf("landslide/%s/input/B%s.tif", i_path, 1:16))) # JUNTAR
      arr = export_h5(rast(img)) # JUNTAR
      path_arr = sprintf("landslide/%s/%s.h5", i_path, i_path)
      if(file.exists(path_arr)) file.remove(path_arr)
      h5write(arr, sprintf("landslide/%s/%s.h5", i_path, i_path), "img")
      
      # 6. Create metadata
      # metadata = create_metadata(point_wgs, i_path)
      # write(metadata, sprintf("landslide/%s/metadata.json", i_path))
    }else{
      vacios = c(vacios, i)
    }
  }, 
  error = function(cond){
    message(i)
    return(NA)
  })
}



img1 = h5read("E:/ai/landslides4sense/TrainData/img/image_1.h5", name = "img")


vacios = c()
for(i in 6:nrow(points)){
  i_path = paste0("point_", str_pad(as.character(i), 3, side = "left", pad = "0"))
  if (length(list.files(sprintf("landslide/%s/input", i_path))) == 0){
    vacios = c(vacios, i)
  }
}


tb = tibble(row = 1:320, 
       existe = !1:320 %in% vacios)
length(tb[tb$existe == T,]$row)

rows = tb[tb$existe == T,]$row


rows[4:length(rows)]

