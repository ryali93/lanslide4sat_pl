library(rgee)
library(tidyverse)
library(sf)
library(raster)
library(terra)
library(jsonlite)
library(rhdf5)

ee_Initialize(drive = T)

setwd("E:/ai/new_dataset")

source("https://gist.githubusercontent.com/ryali93/a909bf1176577143f002d5b3e189b281/raw/de89c77fbfa67d0e63e9e9f5d531710cd05ca923/get_data_utils.R")

# ---------------------------------------------------------------

path_points <- "E:/ai/desliza_f.kml"
# Create folders
create_dir_general()
# Read points
points <- read_point(path_points)

vacios = c()
for(i in 1:nrow(points)){
  i_path = paste0("point_", str_pad(as.character(i), 3, side = "left", pad = "0"))
  print(i_path)
  tryCatch({
      # 1. Create dirs and save points
      point = points[i,]
      point_wgs = st_transform(point, 4326)
      create_dir_point(i_path)
      st_write(point, dsn = sprintf("landslide/%s/%s.gpkg", i_path, i_path), layer = i_path)
      
      # 2. Create polygons
      polygon = generate_patch(point, 1490, 1490)
      polygon_ee = create_pol(polygon)
      
      # 3. Download satellite images
      srtm = download_srtm(polygon_ee, res=10)
      sen2 = download_sen2(polygon_ee, res=10)
      img = merge_img(sen2, srtm)
      
      # 4. Save imgs
      for(i in 1:nlayers(img)){
        terra::writeRaster(img[[i]], sprintf("landslide/%s/input/B%s.tif", i_path, as.character(i)), overwrite=T)
      }
      
      # 5. Export h5 file
      arr = export_h5(rast(img)) # JUNTAR
      path_arr = sprintf("landslide/%s/%s.h5", i_path, i_path)
      if(file.exists(path_arr)) file.remove(path_arr)
      h5write(arr, sprintf("landslide/%s/%s.h5", i_path, i_path), "img")
      
      # 6. Create metadata
      metadata = create_metadata(point_wgs, i_path)
      write(metadata, sprintf("landslide/%s/metadata.json", i_path))
  }, 
  error = function(cond){
    message(i)
    return(NA)
  })
}

# 13, 14 -> Alos palsar
# 15, 16 -> descargado a 10m, luego slope
# 17, 18 -> descargado a 30m, luego slope y luego resample a 10m
# 19, 20 -> agregar slope como banda, descargar a 30m y luego resample a 10
# 21, 22 -> agregar slope como banda, descargar a 10m y luego cambio de coords
