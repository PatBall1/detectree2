# necessary basic libraries
import pandas as pd
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
import png
import math
import glob

# geospatial libraries
import rasterio
import geopandas
from geopandas.tools import sjoin
import fiona
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs
import descartes


if __name__ == "__main__":
    ### Right let's test this first with Sepilok 10cm resolution, then I need to try it with 50cm resolution.

    # Read in the tiff file
    data = rasterio.open(
        "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_benchmark/P4_Ortho_2015.tif"
    )
    resolution = data.rasterio.resolution()
    # Gonna need to do this!
    # Read in shapefile of crowns, if training on your own data!
    # crowns = geopandas.read_file('/home/jovyan/lustre_scratch/sepilok_data/sep_danum_crowns_no_overlap/all_manual_crowns_no_overlap.shp')

    # let's investigate the tiff, what is the shape? Bounds? Bands? CRS?
    # show a plot of it too

    print(
        "shape =",
        data.shape,
        ",",
        data.bounds,
        "and number of bands =",
        data.count,
        ", crs =",
        data.crs,
    )

    # have a look if you want (usually slow)
    # show(data)

    ## find x and y origin of the tiff, need to set the north western corner as the origin

    print(data.bounds)
    tiff_x_origin = data.bounds[0]
    tiff_y_origin = data.bounds[3]
    print("Tiff x origin:", tiff_x_origin)
    print("Tiff y origin:", tiff_y_origin)

    ### First let's read in all the bands of this tiff
    ### this is naturally slow...
    ### and we run out of memory...so maybe we should lazily load with a new approach...rioxarray

    ### SO I NEED TO THINK ABOUT DIAGONALS HERE
    ### HOW DO ORIGINS WORK FOR DIAGONALS
    ### IS THE PNG JUST THE CUT OUT OF THE TILE
    ### OR DOES IT INCLUDE THE WHITE SPACE AROUND THE TILE

    ### It seems like it is just saving them all as squares...which would be great...not sure how to corroborate this other than inspection by eye...bounding box perhaps?

    arr = data.read()

    R = arr[0]
    G = arr[1]
    B = arr[2]

    # stack up the bands in an order appropriate for saving with cv2, then rescale to the correct 0-255 range for cv2

    rgb = np.dstack((B, G, R))  # BGR for cv2

    rgb_rescaled = 255 * (
        rgb / np.amax(rgb)
    )  # scale the values of the bands if they are non-standard to range 0-255

    # rgb_rescaled = rgb # usually rescaling is not required, but it depends on your tiff

    # save this as jpg or png...we are going for png...again, named with the origin of the specific tile
    cv2.imwrite(
        "/content/drive/Shareddrives/detectreeRGB/paracou/full_png/Paracou_full_site.png",
        rgb_rescaled,
    )

    ### Ok, let's read in this png...this filename has changed

    im = cv2.imread(
        "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_pngs/P4.png"
    )

    ### Ok, so now I want to tile up this massive png.

    import cv2
    import math

    img = cv2.imread(
        "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_pngs/P4.png"
    )  # 512x512

    img_shape = img.shape
    tile_size = (1400, 1400)
    offset = (1000, 1000)

    for i in np.arange(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
        for j in np.arange(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
            cropped_img = img[
                offset[1] * i : min(offset[1] * i + tile_size[1], img_shape[0]),
                offset[0] * j : min(offset[0] * j + tile_size[0], img_shape[1]),
            ]
            # Debugging the tiles
            cv2.imwrite(
                "/content/drive/Shareddrives/detectreeRGB/benchmark/tiled_pngs3/"
                + str(i)
                + "_"
                + str(j)
                + ".png",
                cropped_img,
            )
