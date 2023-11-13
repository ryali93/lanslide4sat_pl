library(rgee)
library(tidyverse)
library(sf)
library(raster)
library(terra)
library(jsonlite)
library(rhdf5)

# Initialize Google Earth Engine with Google Drive access
ee_Initialize(drive = TRUE)

# Set the working directory
source("get_data_utils.R")
setwd("~/l4s/dataset")

path_points <- "points/points.gpkg" # Path to the geographic points file
create_dir_general() # Create general directories for data storage
points <- read_point(path_points) # Read points from the geographic points file
vacios = c()

for(i in 1:nrow(points)){
  i_path = paste0("point_", str_pad(as.character(i), 3, side = "left", pad = "0"))
  print(i_path)
  tryCatch({
      # 1. Create directories for each point and save the points
      point = points[i,]
      point_wgs = st_transform(point, 4326)
      create_dir_point(i_path)
      st_write(point, dsn = sprintf("l4s/%s/%s.gpkg", i_path, i_path), layer = i_path)
      
      # 2. Generate polygons around the points
      polygon = generate_patch(point, 1490, 1490)
      polygon_ee = create_pol(polygon)
      
      # 3. Download satellite images (SRTM and Sentinel-2)
      srtm = download_srtm(polygon_ee, res=10)
      sen2 = download_sen2(polygon_ee, res=10)
      img = merge_img(sen2, srtm)
      
      # 4. Save raster images in TIF format
      for(i in 1:nlayers(img)){
        terra::writeRaster(img[[i]], sprintf("l4s/%s/input/B%s.tif", i_path, as.character(i)), overwrite=TRUE)
      }
      
      # 5. Export data to an H5 file
      arr = export_h5(rast(img))
      path_arr = sprintf("l4s/%s/%s.h5", i_path, i_path)
      if(file.exists(path_arr)) file.remove(path_arr)
      h5write(arr, sprintf("l4s/%s/%s.h5", i_path, i_path), "img")
      
      # 6. Create and save metadata
      metadata = create_metadata(point_wgs, i_path)
      write(metadata, sprintf("l4s/%s/metadata.json", i_path))
  }, 
  error = function(cond){
    message(i)
    return(NA)
  })
}
