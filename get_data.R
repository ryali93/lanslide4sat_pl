library(rlist)
library(rgee)
library(mapview)
library(sf)

ee_Initialize()

d = st_read("E:/ai/desliza_s.kml")
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
  
  # pol_p = sf::st_sfc(pol, crs = 32718)
  return(pol)
}

lista_p = list()
for(i in 1:5){
  p = patch(i, 1280, 1280)
  lista_p = list.append(lista_p, p)
}
pols = st_sfc(lista_p, crs=32718)
mapview(list(d_t, pols))
