"""Tiling orthomosaic and crown data.

These functions tile orthomosaics and crown data for training and evaluation
of models and making landscape predictions.
"""

import concurrent.futures
import json
import logging
import os
import pickle
import random
import shutil
import warnings  # noqa: F401
from math import ceil
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from fiona.crs import from_epsg  # noqa: F401
# from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
# from rasterio.io import DatasetReader
from rasterio.mask import mask
# from rasterio.windows import from_bounds
from shapely.geometry import box

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def load_class_mapping(file_path: str):
    """Function to load class-to-index mapping from a file.

    Args:
        file_path: Path to the file (json or pickle)

    Returns:
        class_to_idx: Loaded class-to-index mapping
    """
    file_ext = Path(file_path).suffix

    if file_ext == '.json':
        with open(file_path, 'r') as f:
            class_to_idx = json.load(f)
    elif file_ext == '.pkl':
        with open(file_path, 'rb') as f:
            class_to_idx = pickle.load(f)
    else:
        raise ValueError("Unsupported file format. Use '.json' or '.pkl'.")

    return class_to_idx


def process_tile(
    img_path: str,
    out_dir: str,
    buffer: int,
    tile_width: int,
    tile_height: int,
    dtype_bool: bool,
    minx,
    miny,
    crs,
    tilename,
    crowns: gpd.GeoDataFrame = None,
    threshold: float = 0,
    nan_threshold: float = 0,
):
    """Process a single tile for making predictions.

    Args:
        img_path: Path to the orthomosaic
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
    try:
        with rasterio.open(img_path) as data:
            out_path = Path(out_dir)
            out_path_root = out_path / f"{tilename}_{minx}_{miny}_{tile_width}_{buffer}_{crs}"

            minx_buffered = minx - buffer
            miny_buffered = miny - buffer
            maxx_buffered = minx + tile_width + buffer
            maxy_buffered = miny + tile_height + buffer

            bbox = box(minx_buffered, miny_buffered, maxx_buffered, maxy_buffered)
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=data.crs)
            coords = get_features(geo)

            if crowns is not None:
                overlapping_crowns = gpd.clip(crowns, geo)
                if overlapping_crowns.empty or (overlapping_crowns.dissolve().area[0] / geo.area[0]) < threshold:
                    return None
            else:
                overlapping_crowns = None

            out_img, out_transform = mask(data, shapes=coords, crop=True)

            out_sumbands = np.sum(out_img, axis=0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.where(out_sumbands == 765, 1, 0)
            sumzero = zero_mask.sum()
            sumnan = nan_mask.sum()
            totalpix = out_img.shape[1] * out_img.shape[2]

            # If the tile is mostly empty or mostly nan, don't save it
            if sumzero > nan_threshold * totalpix or sumnan > nan_threshold * totalpix:
                return None

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

            out_tif = out_path_root.with_suffix(".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            with rasterio.open(out_tif) as clipped:
                arr = clipped.read()
                r, g, b = arr[0], arr[1], arr[2]
                rgb = np.dstack((b, g, r))  # Reorder for cv2 (BGRA)

                # Rescale to 0-255 if necessary
                if np.max(g) > 255:
                    rgb_rescaled = 255 * rgb / 65535
                else:
                    rgb_rescaled = rgb

                cv2.imwrite(str(out_path_root.with_suffix(".png").resolve()), rgb_rescaled)

            if overlapping_crowns is not None:
                return data, out_path_root, overlapping_crowns, minx, miny, buffer

            return data, out_path_root, None, minx, miny, buffer

    except RasterioIOError as e:
        logger.error(f"RasterioIOError while applying mask {coords}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing tile {tilename} at ({minx}, {miny}): {e}")
        return None


def process_tile_ms(
    img_path: str,
    out_dir: str,
    buffer: int,
    tile_width: int,
    tile_height: int,
    dtype_bool: bool,
    minx,
    miny,
    crs,
    tilename,
    crowns: gpd.GeoDataFrame = None,
    threshold: float = 0,
    nan_threshold: float = 0,
):
    """Process a single tile for making predictions.

    Args:
        img_path: Path to the orthomosaic
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
    try:
        with rasterio.open(img_path) as data:
            out_path = Path(out_dir)
            out_path_root = out_path / f"{tilename}_{minx}_{miny}_{tile_width}_{buffer}_{crs}"

            minx_buffered = minx - buffer
            miny_buffered = miny - buffer
            maxx_buffered = minx + tile_width + buffer
            maxy_buffered = miny + tile_height + buffer

            bbox = box(minx_buffered, miny_buffered, maxx_buffered, maxy_buffered)
            geo = gpd.GeoDataFrame({"geometry": [bbox]}, index=[0], crs=data.crs)
            coords = [geo.geometry[0].__geo_interface__]

            if crowns is not None:
                overlapping_crowns = gpd.clip(crowns, geo)
                if overlapping_crowns.empty or (overlapping_crowns.dissolve().area[0] / geo.area[0]) < threshold:
                    return None
            else:
                overlapping_crowns = None

            out_img, out_transform = mask(data, shapes=coords, crop=True)

            out_sumbands = np.sum(out_img, axis=0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.isnan(out_sumbands)
            sumzero = zero_mask.sum()
            sumnan = nan_mask.sum()
            totalpix = out_img.shape[1] * out_img.shape[2]

            # If the tile is mostly empty or mostly nan, don't save it
            if sumzero > nan_threshold * totalpix or sumnan > nan_threshold * totalpix:
                return None

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

            out_tif = out_path_root.with_suffix(".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            if overlapping_crowns is not None:
                return data, out_path_root, overlapping_crowns, minx, miny, buffer

            return data, out_path_root, None, minx, miny, buffer

    except RasterioIOError as e:
        logger.error(f"RasterioIOError while applying mask {coords}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing tile {tilename} at ({minx}, {miny}): {e}")
        return None


def process_tile_train(
    img_path: str,
    out_dir: str,
    buffer: int,
    tile_width: int,
    tile_height: int,
    dtype_bool: bool,
    minx,
    miny,
    crs,
    tilename,
    crowns: gpd.GeoDataFrame,
    threshold,
    nan_threshold,
    mode: str = "rgb",
    class_column: str = None,  # Allow user to specify class column
) -> None:
    """Process a single tile for training data.

    Args:
        img_path: Path to the orthomosaic
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
    if mode == "rgb":
        result = process_tile(img_path, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs,
                              tilename, crowns, threshold, nan_threshold)
    elif mode == "ms":
        result = process_tile_ms(img_path, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs,
                                 tilename, crowns, threshold, nan_threshold)

    if result is None:
        # logger.warning(f"Skipping tile at ({minx}, {miny}) due to insufficient data.")
        return

    data, out_path_root, overlapping_crowns, minx, miny, buffer = result

    if overlapping_crowns is not None and not overlapping_crowns.empty:
        overlapping_crowns = overlapping_crowns.explode(index_parts=True)
        moved = overlapping_crowns.translate(-minx + buffer, -miny + buffer)
        scalingx = 1 / (data.transform[0])
        scalingy = -1 / (data.transform[4])
        moved_scaled = moved.scale(scalingx, scalingy, origin=(0, 0))

        if mode == "rgb":
            impath = {"imagePath": out_path_root.with_suffix(".png").as_posix()}
        elif mode == "ms":
            impath = {"imagePath": out_path_root.with_suffix(".tif").as_posix()}

        try:
            filename = out_path_root.with_suffix(".geojson")
            moved_scaled = overlapping_crowns.set_geometry(moved_scaled)

            if class_column is not None:
                # Ensure we map the selected column to the 'status' field
                moved_scaled['status'] = moved_scaled[class_column]
                # Keep only 'status' and geometry
                moved_scaled = moved_scaled[['geometry', 'status']]
            else:
                # Keep only geometry to reduce file size
                moved_scaled = moved_scaled[['geometry']]

            # Save the result as GeoJSON
            moved_scaled.to_file(driver="GeoJSON", filename=filename)

            # Add image path info to the GeoJSON file
            with open(filename, "r") as f:
                shp = json.load(f)
                shp.update(impath)
            with open(filename, "w") as f:
                json.dump(shp, f)
        except ValueError:
            logger.warning("Cannot write empty DataFrame to file.")
            return
    else:
        return None  # Handle the case where there are no overlapping crowns


# Define a top-level helper function
def process_tile_train_helper(args):
    return process_tile_train(*args)


def tile_data(
    img_path: str,
    out_dir: str,
    buffer: int = 30,
    tile_width: int = 200,
    tile_height: int = 200,
    crowns: gpd.GeoDataFrame = None,
    threshold: float = 0,
    nan_threshold: float = 0.1,
    dtype_bool: bool = False,
    mode: str = "rgb",
    class_column: str = None,  # Allow class column to be passed here
) -> None:
    """Tiles up orthomosaic and corresponding crowns (if supplied) into training/prediction tiles.

    Tiles up large rasters into managable tiles for training and prediction. If crowns are not supplied the function
    will tile up the entire landscape for prediction. If crowns are supplied the function will tile these with the image
    and skip tiles without a minimum coverage of crowns. The 'threshold' can be varied to ensure a good coverage of
    crowns across a traing tile. Tiles that do not have sufficient coverage are skipped.

    Args:
        img_path: Path to the orthomosaic
        out_dir: Output directory
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
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)
    tilename = Path(img_path).stem
    with rasterio.open(img_path) as data:
        crs = data.crs.to_epsg()  # Update CRS handling to avoid deprecated syntax

        tile_args = [
            (img_path, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs, tilename, crowns,
             threshold, nan_threshold, mode, class_column)
            for minx in np.arange(ceil(data.bounds[0]) + buffer, data.bounds[2] - tile_width - buffer, tile_width, int)
            for miny in np.arange(ceil(data.bounds[1]) + buffer, data.bounds[3] - tile_height - buffer, tile_height,
                                  int)
        ]

        with concurrent.futures.ProcessPoolExecutor() as executor:  # Use ProcessPoolExecutor here
            list(executor.map(process_tile_train_helper, tile_args))

    logger.info("Tiling complete")


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


def record_classes(crowns: gpd.GeoDataFrame, out_dir: str, column: str = 'status', save_format: str = 'json'):
    """Function that records a list of classes into a file that can be read during training.

    Args:
        crowns: gpd dataframe with the crowns
        out_dir: directory to save the file
        column: column name to get the classes from
        save_format: format to save the file ('json' or 'pickle')

    Returns:
        None
    """
    # Extract unique class names from the specified column
    list_of_classes = crowns[column].unique().tolist()

    # Sort the list of classes in alphabetical order
    list_of_classes.sort()

    # Create a dictionary for class-to-index mapping
    class_to_idx = {class_name: idx for idx, class_name in enumerate(list_of_classes)}

    # Save the class-to-index mapping to disk
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)

    if save_format == 'json':
        with open(out_path / 'class_to_idx.json', 'w') as f:
            json.dump(class_to_idx, f)
    elif save_format == 'pickle':
        with open(out_path / 'class_to_idx.pkl', 'wb') as f:
            pickle.dump(class_to_idx, f)
    else:
        raise ValueError("Unsupported save format. Use 'json' or 'pickle'.")

    print(f"Classes saved as {save_format} file: {class_to_idx}")


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

    # file_names = tiles_dir.glob("*.png")
    file_names = tiles_dir.glob("*.geojson")
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
    # Define paths to the input data
    img_path = "/path/to/your/orthomosaic.tif"  # Path to your input orthomosaic file
    crown_path = "/path/to/your/crown_shapefile.shp"  # Path to the shapefile containing crowns
    out_dir = "/path/to/output/directory"  # Directory where you want to save the tiled output

    # Optional parameters for tiling and processing
    buffer = 30  # Overlap between tiles (in meters)
    tile_width = 200  # Tile width (in meters)
    tile_height = 200  # Tile height (in meters)
    nan_threshold = 0.1  # Max proportion of tile that can be NaN before it's discarded
    threshold = 0.5  # Minimum crown coverage per tile for it to be kept (0-1)
    dtype_bool = False  # Change dtype to uint8 to avoid black tiles
    mode = "rgb"  # Use 'rgb' for regular 3-channel imagery, 'ms' for multispectral
    class_column = "species"  # Column in the crowns file to use as the class label

    # Read in the crowns
    crowns = gpd.read_file(crown_path)

    # Record the classes and save the class mapping
    record_classes(
        crowns=crowns,  # Geopandas dataframe with crowns
        out_dir=out_dir,  # Output directory to save class mapping
        column=class_column,  # Column used for classes
        save_format='json'  # Choose between 'json' or 'pickle'
    )

    # Perform the tiling, ensuring the selected class column is used
    tile_data(
        img_path=img_path,
        out_dir=out_dir,
        buffer=buffer,
        tile_width=tile_width,
        tile_height=tile_height,
        crowns=crowns,
        threshold=threshold,
        nan_threshold=nan_threshold,
        dtype_bool=dtype_bool,
        mode=mode,
        class_column=class_column  # Use the selected class column (e.g., 'species', 'status')
    )

    # Split the data into training and validation sets (optional)
    # This can be used for train/test folder creation based on the generated tiles
    to_traintest_folders(
        tiles_folder=out_dir,  # Directory where tiles are saved
        out_folder="/path/to/final/data/output",  # Final directory for train/test data
        test_frac=0.2,  # Fraction of data to be used for testing
        folds=5,  # Number of folds (optional, can be set to 1 for no fold splitting)
        strict=True,  # Ensure no overlap between train/test tiles
        seed=42  # Set seed for reproducibility
    )

    logger.info("Tiling process completed successfully!")
