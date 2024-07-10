"""Tiling orthomosaic and crown data.

These functions tile orthomosaics and crown data for training and evaluation
of models and making landscape predictions.
"""

import concurrent.futures
import json
import os
import random
import shutil
import warnings
from math import ceil
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from fiona.crs import from_epsg  # noqa: F401
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.mask import mask
from shapely.geometry import box

# class img_data(DatasetReader):
#    """
#    Class for image data to be processed for tiling
#    """
#
#    def __init__(self):
#        self.x_origin = self.bounds[0]
#        self.y_origin = self.bounds[3]
#        self.pixelSizeX = self.affine[0]
#        self.pixelSizeY = -self.affine[4]
#


def get_features(gdf: gpd.GeoDataFrame):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them.

    Args:
      gdf: Input geopandas dataframe

    Returns:
      json style data
    """
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def process_tile(
    data: DatasetReader,
    out_dir: str,
    buffer: int,
    tile_width: int,
    tile_height: int,
    dtype_bool: bool,
    minx,
    miny,
    crs,
    tilename
) -> None:
    """Process a single tile for making predictions.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        out_dir: Output directory
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        dtype_bool: Flag to edit dtype to prevent black tiles
        minx: Minimum x coordinate of tile
        miny: Minimum y coordinate of tile
        crs: Coordinate reference system
        tilename: Name of the tile
    
    Returns:
        None
    """
    out_path = Path(out_dir)
    out_path_root = out_path / f"{tilename}_{minx}_{miny}_{tile_width}_{buffer}_{crs}"
    bbox = box(minx - buffer, miny - buffer, minx + tile_width + buffer, miny + tile_height + buffer)
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
    coords = get_features(geo)
    out_img, out_transform = mask(data, shapes=coords, crop=True)

    out_sumbands = np.sum(out_img, 0)
    zero_mask = np.where(out_sumbands == 0, 1, 0)
    nan_mask = np.where(out_sumbands == 765, 1, 0)
    sumzero = zero_mask.sum()
    sumnan = nan_mask.sum()
    totalpix = out_img.shape[1] * out_img.shape[2]

    # If the tile is mostly empty or mostly nan, don't save it
    if sumzero > 0.25 * totalpix or sumnan > 0.25 * totalpix:
        return

    out_meta = data.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "nodata": None,
    })
    if dtype_bool:
        out_meta.update({"dtype": "uint8"})

    out_tif = out_path_root.with_suffix(out_path_root.suffix + ".tif")
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)

    clipped = rasterio.open(out_tif)
    arr = clipped.read()
    r, g, b = arr[0], arr[1], arr[2]
    rgb = np.dstack((b, g, r))

    # Rescale to 0-255 if necessary
    if np.max(g) > 255:
        rgb_rescaled = 255 * rgb / 65535
    else:
        rgb_rescaled = rgb

    cv2.imwrite(str(out_path_root.with_suffix(out_path_root.suffix + ".png").resolve()), rgb_rescaled)


def tile_data(
    data: DatasetReader,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    dtype_bool: bool = False,
) -> None:
    """Tiles up orthomosaic for making predictions on.

    Tiles up full othomosaic into managable chunks to make predictions on. Use tile_data_train to generate tiled
    training data. A bug exists on some input raster types whereby outputed tiles are completely black - the dtype_bool
    argument should be switched if this is the case.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        dtype_bool: Flag to edit dtype to prevent black tiles

    Returns:
        None
    """
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)
    crs = CRS.from_string(data.crs.wkt).to_epsg()
    tilename = Path(data.name).stem
    total_tiles = int(((data.bounds[2] - data.bounds[0]) / tile_width) * ((data.bounds[3] - data.bounds[1]) / tile_height))

    tile_args = [
        (data, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs, tilename)
        for minx in np.arange(data.bounds[0], data.bounds[2] - tile_width, tile_width, int)
        for miny in np.arange(data.bounds[1], data.bounds[3] - tile_height, tile_height, int)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(lambda args: process_tile(*args), tile_args))

    print("Tiling complete")


def process_tile_train(
    data: DatasetReader,
    out_dir: str,
    buffer: int,
    tile_width: int,
    tile_height: int,
    dtype_bool: bool,
    minx,
    miny,
    crs,
    tilename,
    crowns,
    threshold,
    nan_threshold
) -> None:
    """Process a single tile for training data.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        out_dir: Output directory
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        dtype_bool: Flag to edit dtype to prevent black tiles
        minx: Minimum x coordinate of tile
        miny: Minimum y coordinate of tile
        crs: Coordinate reference system
        tilename: Name of the tile
        crowns: Crown polygons as a geopandas dataframe
        threshold: Min proportion of the tile covered by crowns to be accepted {0,1}
        nan_theshold: Max proportion of tile covered by nans
    
    Returns:
        None
    """

    out_path = Path(out_dir)
    out_path_root = out_path / f"{tilename}_{minx}_{miny}_{tile_width}_{buffer}_{crs}"

    minx_buffered = minx - buffer
    miny_buffered = miny - buffer
    maxx_buffered = minx + tile_width + buffer
    maxy_buffered = miny + tile_height + buffer

    bbox = box(minx_buffered, miny_buffered, maxx_buffered, maxy_buffered)
    geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
    coords = get_features(geo)

    overlapping_crowns = gpd.clip(crowns, geo)
    if overlapping_crowns.empty or (overlapping_crowns.dissolve().area[0] / geo.area[0]) < threshold:
        return

    out_img, out_transform = mask(data, shapes=coords, crop=True)
    out_sumbands = np.sum(out_img, 0)
    zero_mask = np.where(out_sumbands == 0, 1, 0)
    nan_mask = np.where(out_sumbands == 765, 1, 0)
    sumzero = zero_mask.sum()
    sumnan = nan_mask.sum()
    totalpix = out_img.shape[1] * out_img.shape[2]
    # If the tile is mostly empty or mostly nan, don't save it
    if sumzero > nan_threshold * totalpix or sumnan > nan_threshold * totalpix:
        return

    out_meta = data.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "nodata": None,
    })
    if dtype_bool:
        out_meta.update({"dtype": "uint8"})

    out_tif = out_path_root.with_suffix(out_path_root.suffix + ".tif")
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)

    clipped = rasterio.open(out_tif)
    arr = clipped.read()
    r, g, b = arr[0], arr[1], arr[2]
    rgb = np.dstack((b, g, r))

    # Rescale if necessary
    if np.max(g) > 255:
        rgb_rescaled = 255 * rgb / 65535
    else:
        rgb_rescaled = rgb

    cv2.imwrite(str(out_path_root.with_suffix(out_path_root.suffix + ".png").resolve()), rgb_rescaled)

    moved = overlapping_crowns.translate(-minx + buffer, -miny + buffer)
    scalingx = 1 / (data.transform[0])
    scalingy = -1 / (data.transform[4])
    moved_scaled = moved.scale(scalingx, scalingy, origin=(0, 0))
    impath = {"imagePath": out_path_root.with_suffix(out_path_root.suffix + ".png").as_posix()}

    try:
        filename = out_path_root.with_suffix(out_path_root.suffix + ".geojson")
        moved_scaled = overlapping_crowns.set_geometry(moved_scaled)
        moved_scaled.to_file(driver="GeoJSON", filename=filename)
        with open(filename, "r") as f:
            shp = json.load(f)
            shp.update(impath)
        with open(filename, "w") as f:
            json.dump(shp, f)
    except ValueError:
        print("Cannot write empty DataFrame to file.")
        return

def tile_data_train(  # noqa: C901
    data: DatasetReader,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    crowns: gpd.GeoDataFrame = None,
    threshold: float = 0,
    nan_threshold: float = 0.1,
    dtype_bool: bool = False,
) -> None:
    """Tiles up orthomosaic and corresponding crowns into training tiles.

    A threshold can be used to ensure a good coverage of crowns across a tile. Tiles that do not have sufficient
    coverage are rejected.

    Args:
        data: Orthomosaic as a rasterio object in a UTM type projection
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        crowns: Crown polygons as a geopandas dataframe
        threshold: Min proportion of the tile covered by crowns to be accepted {0,1}
        nan_theshold: Max proportion of tile covered by nans
        dtype_bool: Flag to edit dtype to prevent black tiles

    Returns:
        None

    """

    # TODO: Clip data to crowns straight away to speed things up
    # TODO: Tighten up epsg handling
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)
    tilename = Path(data.name).stem
    crs = CRS.from_string(data.crs.wkt).to_epsg()

    tile_args = [
        (data, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs, tilename, crowns, threshold, nan_threshold)
        for minx in np.arange(ceil(data.bounds[0]) + buffer, data.bounds[2] - tile_width - buffer, tile_width, int)
        for miny in np.arange(ceil(data.bounds[1]) + buffer, data.bounds[3] - tile_height - buffer, tile_height, int)
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(lambda args: process_tile_train(*args), tile_args))

    print("Tiling complete")


def image_details(fileroot):
    """Take a filename and split it up to get the coordinates, tile width and the buffer and then output box structure.

    Args:
        fileroot: image filename without file extension

    Returns:
        Box structure
    """
    image_info = fileroot.split("_")
    minx = int(image_info[-5])
    miny = int(image_info[-4])
    tile_width = int(image_info[-3])
    buffer = int(image_info[-2])

    xbox_coords = (minx - buffer, minx + tile_width + buffer)
    ybox_coords = (miny - buffer, miny + tile_width + buffer)
    return [xbox_coords, ybox_coords]


def is_overlapping_box(test_boxes_array, train_box):
    """Check if the train box overlaps with any of the test boxes.

    Args:
        test_boxes_array:
        train_box:

    Returns:
        Boolean
    """
    for test_box in test_boxes_array:
        test_box_x = test_box[0]
        test_box_y = test_box[1]
        train_box_x = train_box[0]
        train_box_y = train_box[1]

        # Check if both the x and y coords overlap meaning the entire box does and hence end loop
        if test_box_x[1] > train_box_x[0] and train_box_x[1] > test_box_x[0]:
            if test_box_y[1] > train_box_y[0] and train_box_y[1] > test_box_y[0]:
                return True

    return False


def record_data(crowns,
                out_dir,
                column='status'):
    """Function that will record a list of classes into a file that can be readed during training.

    Args:
        crowns: gpd dataframe with the crowns
        out_dir: directory to save the file
        column: column name to get the classes from

    Returns:
        None
    """

    list_of_classes = crowns[column].unique().tolist()

    print("**The list of classes are:**")
    print(list_of_classes)
    print("**The list has been saved to the out_dir**")

    # Write it into file "classes.txt"
    out_tif = out_dir + 'classes.txt'
    f = open(out_tif, "w")
    for i in list_of_classes:
        f.write("%s\n" % i)
    f.close()


def to_traintest_folders(  # noqa: C901
        tiles_folder: str = "./",
        out_folder: str = "./data/",
        test_frac: float = 0.2,
        folds: int = 1,
        strict: bool = False,
        seed: int = None) -> None:
    """Send tiles to training (+validation) and test dir

    With "strict" it is possible to automatically ensure no overlap between train/val and test tiles.

    Args:
        tiles_folder: folder with tiles
        out_folder: folder to save train and test folders
        test_frac: fraction of tiles to be used for testing
        folds: number of folds to split the data into
        strict: if True, training/validation files will be removed if there is any overlap with test files (inc buffer)

    Returns:
        None
    """
    tiles_dir = Path(tiles_folder)
    out_dir = Path(out_folder)

    if not os.path.exists(tiles_dir):
        raise IOError

    if Path(out_dir / "train").exists() and Path(out_dir / "train").is_dir():
        shutil.rmtree(Path(out_dir / "train"))
    if Path(out_dir / "test").exists() and Path(out_dir / "test").is_dir():
        shutil.rmtree(Path(out_dir / "test"))
    Path(out_dir / "train").mkdir(parents=True, exist_ok=True)
    Path(out_dir / "test").mkdir(parents=True, exist_ok=True)

    file_names = tiles_dir.glob("*.png")
    file_roots = [item.stem for item in file_names]

    num = list(range(0, len(file_roots)))

    # this affects the random module module-wide
    if seed is not None:
        random.seed(seed)
    random.shuffle(num)

    test_boxes = []

    for i in range(0, len(file_roots)):
        # copy to test
        if i < len(file_roots) * test_frac:
            test_boxes.append(image_details(file_roots[num[i]]))
            shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(
                Path(file_roots[num[i]]).suffix + ".geojson"), out_dir / "test")
        else:
            # copy to train
            train_box = image_details(file_roots[num[i]])
            if strict:   # check if there is overlap with test boxes
                if not is_overlapping_box(test_boxes, train_box):
                    shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(
                        Path(file_roots[num[i]]).suffix + ".geojson"), out_dir / "train")
            else:
                shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(
                    Path(file_roots[num[i]]).suffix + ".geojson"), out_dir / "train")

    # COMMENT NECESSARY HERE
    file_names = (out_dir / "train").glob("*.geojson")
    file_roots = [item.stem for item in file_names]
    # stemname = Path(filenames[0]).stem.split("_", 1)[0]
    # indices = [item.split("_", 1)[-1].split(".", 1)[0] for item in filenames]
    # random.shuffle(indices)
    num = list(range(0, len(file_roots)))
    random.shuffle(num)
    ind_split = np.array_split(file_roots, folds)

    for i in range(0, folds):
        Path(out_dir / f"train/fold_{i + 1}").mkdir(parents=True, exist_ok=True)
        for name in ind_split[i]:
            shutil.move(
                out_dir / f"train/{name}.geojson",  # type: ignore
                out_dir / f"train/fold_{i + 1}/{name}.geojson",
            )


if __name__ == "__main__":
    # Right let"s test this first with Sepilok 10cm resolution, then I need to try it with 50cm resolution.
    img_path = "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_benchmark/P4_Ortho_2015.tif"
    crown_path = "gdrive/MyDrive/JamesHirst/NY/Buffalo/Buffalo_raw_data/all_crowns.shp"
    out_dir = "./"
    # Read in the tiff file
    # data = img_data.open(img_path)
    # Read in crowns
    data = rasterio.open(img_path)
    crowns = gpd.read_file(crown_path)
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

    buffer = 20
    tile_width = 200
    tile_height = 200

    tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns)
    to_traintest_folders(folds=5)
