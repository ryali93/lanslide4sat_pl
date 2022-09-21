library(rlist)
library(rgee)
library(mapview)
library(sf)
library(raster)
library(terra)
library(rhdf5)

ee_Initialize(drive = T)

d = st_read("E:/ai/desliza_f.kml")
d = st_zm(d, drop = TRUE, what = "ZM")
d_t = st_transform(d, crs = 32718)

lista_p = list()
patch = function(i, w, h){
  x = st_coordinates(d_t[i,])[1]
  y = st_coordinates(d_t[i,])[2]
  w_m = w/2
  h_m = h/2
  
  pol = st_polygon(
    list(cbind(
      c(x - w_m, x + w_m, x + w_m, x - w_m, x - w_m),
      c(y + h_m, y + h_m, y - h_m, y - h_m, y + h_m)))
  )
  
  return(pol)
}

lista_p = list()
for(i in 1:5){
  p = patch(i, 1270, 1270)
  lista_p = list.append(lista_p, p)
}
pols = st_sfc(lista_p, crs=32718)
pols_wgs = st_transform(pols, 4326)

p1 = sf_as_ee(pols_wgs[[1]])

# DOWNLOAD SENTINEL2 IMAGES 
dataset = ee$ImageCollection('COPERNICUS/S2_SR')$
  filterDate('2018-01-01', '2020-01-30')$
  filter(ee$Filter$lt('CLOUDY_PIXEL_PERCENTAGE', 10))$
  filterBounds(p1)$
  first()$
  select("B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12")

d1 = ee_as_raster(image=dataset, region=p1, via = "drive", scale = 10)

# DOWNLOAD ALOS PALSAR IMAGES
r1 = raster("E:/ai/nuevo/AP_25513_PLR_F6960_RT1/AP_25513_PLR_F6960_RT1.dem.tif")
r1_clip = crop(r1, as(pols[[1]], "Spatial"))
slope1_clip = slopeAspect(r1_clip, out="slope")


# MERGE DATASET
B1_12 = brick("E:/ai/nuevo/01_landslide.tif")
r1_r = raster::resample(r1_clip, B1_12)
s1_r = slopeAspect(r1_r, out="slope")

B1_14 = addLayer(B1_12, r1_r, s1_r)

writeRaster(B1_14, "E:/ai/nuevo/0001.tif", overwrite=T)


# EXPORT TO H5
s = rast("E:/ai/nuevo/0001.tif")
nx = minmax(s)
rn = (s - nx[1,]) / (nx[2,] - nx[1,])

ar = as.array(rn)
writeHDF5Array(ar, "E:/ai/nuevo/0001_ar.tif")

h5write(ar, "E:/ai/nuevo/0001_ar.h5", "img")
