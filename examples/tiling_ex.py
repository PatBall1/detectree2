import os 
from pathlib import Path 
import rasterio
import geopandas as gpd
from detectree2.preprocessing.tiling import tile_data_train, tile_data
from detectree2.preprocessing.tiling import to_traintest_folders

# img_path = "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_benchmark/P4_Ortho_2015.tif"
# crown_path = "gdrive/MyDrive/JamesHirst/NY/Buffalo/Buffalo_raw_data/all_crowns.shp"
# out_dir = "./"

# img_path = "./data/NY/LargeArea_images/naip_hwf/naip_hwf.tif"
# crown_path = "./data/NY/Buffalo_raw_data/all_crowns.shp"
# out_dir = "./output/buffalo/"

img_path = "./data/paracou/Paracou_RGB_2016_10cm.tif"
# crown_path = "./data/paracou/UpdatedCrowns4.gpkg"
crown_path = "./data/paracou/220619_AllSpLabelled.gpkg"
out_root = "./output2/paracou/"

tiling_out_dir = os.path.join(out_root, "tiling") # at present code requires this to have a trailing slash

Path(out_root).mkdir(parents=True, exist_ok=True)
Path(tiling_out_dir).mkdir(parents=True, exist_ok=True)

# Read in the tiff file
# data = img_data.open(img_path)
# Read in crowns
data = rasterio.open(img_path)
crowns = gpd.read_file(crown_path)
# crowns = crowns.to_crs(data.crs.data)

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

# # use for NY (cannot get tiling to work)
# buffer = 20
# tile_width = 200
# tile_height = 200

# use for Paracou (with updated crowns)
buffer = 40
tile_width = 30
tile_height = 30
threshold = 0.6
n_folds = 5
# resolution = 0.6 # in metres per pixel - @James Ball can you get this from the tiff?
tile_data_train(data, tiling_out_dir, buffer, tile_width, tile_height, crowns, threshold)
to_traintest_folders(tiles_folder=tiling_out_dir, out_folder=out_root, test_frac=0.2, folds=n_folds)

# we now have n fold 