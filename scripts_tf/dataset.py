import numpy as np
import glob
import h5py

def read_data(path):
    TRAIN_PATH = f"{path}/img/*.h5" # data_val
    TRAIN_MASK = f'{path}/mask/*.h5' # data_val
    all_train = sorted(glob.glob(TRAIN_PATH))
    all_mask = sorted(glob.glob(TRAIN_MASK))
    return all_train, all_mask

def config_data(all_train, all_mask):
    # 15 bandas: 
    #   Banda R
    #   Banda G
    #   Banda B
    #   Banda NDVI
    #   Banda Slope
    #   Banda Elevation
    n_imgs = len(all_train)
    TRAIN_XX = np.zeros((n_imgs, 128, 128, 15)) # Cambiar 564 por la cantidad de datos que se tienen
    TRAIN_YY = np.zeros((n_imgs, 128, 128, 1)) # Cambiar 564 por la cantidad de datos que se tienen
    for i, (img, mask) in enumerate(zip(all_train, all_mask)):
        print(i, img, mask)
        with h5py.File(img) as hdf:
            ls = list(hdf.keys())
            data = np.array(hdf.get('img'))

            # assign 0 for the nan value
            data[np.isnan(data)] = 0.000001

            # to normalize the data 
            # mid_rgb = data[:, :, 1:4].max() / 2.0
            # mid_slope = data[:, :, 12].max() / 2.0
            # mid_elevation = data[:, :, 13].max() / 2.0

            # ndvi calculation
            # data_red = data[:, :, 3]
            # data_nir = data[:, :, 7]
            # data_ndvi = np.divide(data_nir - data_red,np.add(data_nir, data_red))
            
            # final array
            TRAIN_XX[i, :, :, 0] = data[:, :, 0]
            TRAIN_XX[i, :, :, 1] = data[:, :, 1]
            TRAIN_XX[i, :, :, 2] = data[:, :, 2]
            TRAIN_XX[i, :, :, 3] = data[:, :, 3]
            TRAIN_XX[i, :, :, 4] = data[:, :, 4]
            TRAIN_XX[i, :, :, 5] = data[:, :, 5]
            TRAIN_XX[i, :, :, 6] = data[:, :, 6]
            TRAIN_XX[i, :, :, 7] = data[:, :, 7]
            TRAIN_XX[i, :, :, 8] = data[:, :, 8]
    
        with h5py.File(mask) as hdf:
            ls = list(hdf.keys())
            data=np.array(hdf.get('mask'))
            TRAIN_YY[i, :, :, 0] = data

    TRAIN_XX[np.isnan(TRAIN_XX)] = 0.000001
    print(TRAIN_XX.min(), TRAIN_XX.max(), TRAIN_YY.min(), TRAIN_YY.max())
    return TRAIN_XX, TRAIN_YY