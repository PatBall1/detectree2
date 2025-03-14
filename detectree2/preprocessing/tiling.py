"""Tiling orthomosaic and crown data.

These functions tile orthomosaics and crown data for training and evaluation
of models and making landscape predictions.
"""

import concurrent.futures
import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import warnings  # noqa: F401
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import geopandas as gpd
import numpy as np
import rasterio
# from rasterio.windows import from_bounds
import rasterio.features
from fiona.crs import from_epsg  # noqa: F401
# from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
# from rasterio.io import DatasetReader
from rasterio.mask import mask
from shapely.geometry import box
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Getting rid of unnecessary and confusing errors and warnings
class GDALFilter(logging.Filter):

    def filter(self, record):
        return "GDAL signalled an error: err_no=4, msg='" not in record.getMessage()


rasterlogger = logging.getLogger("rasterio._env")
rasterlogger.addFilter(GDALFilter())


class PyogrioFilter(logging.Filter):

    def filter(self, record):
        return "Created " not in record.getMessage()


pyogriologger = logging.getLogger("pyogrio._io")
pyogriologger.addFilter(PyogrioFilter())

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

dtype_map = {
    np.dtype(np.uint8): ("uint8", 0),
    np.dtype(np.uint16): ("uint16", 0),
    np.dtype(np.uint32): ("uint32", 0),
    np.dtype(np.int8): ("int8", 0),
    np.dtype(np.int16): ("int16", 0),
    np.dtype(np.int32): ("int32", 0),
    np.dtype(np.float16): ("float16", 0.0),
    np.dtype(np.float32): ("float32", 0.0),
    np.dtype(np.float64): ("float64", 0.0),
}


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


def process_tile(img_path: str,
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
                 mask_gdf: gpd.GeoDataFrame = None,
                 additional_nodata: List[Any] = [],
                 image_statistics: List[Dict[str, float]] = None,
                 ignore_bands_indices: List[int] = [],
                 use_convex_mask: bool = True):
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
                if mask_gdf is not None:
                    overlapping_crowns = gpd.clip(overlapping_crowns, mask_gdf)
                if overlapping_crowns.empty or (overlapping_crowns.dissolve().area[0] / geo.area[0]) < threshold:
                    return None
            else:
                overlapping_crowns = None

            if data.nodata is not None:
                nodata = data.nodata
            else:
                nodata = 0

            out_img, out_transform = mask(data, shapes=coords, nodata=nodata, crop=True, indexes=[1, 2, 3])

            mask_tif = None
            if mask_gdf is not None:
                #if mask_gdf.crs != data.crs:
                #    mask_gdf = mask_gdf.to_crs(data.crs) #TODO is this necessary?

                mask_tif = rasterio.features.geometry_mask([geom for geom in mask_gdf.geometry],
                                                           transform=out_transform,
                                                           invert=True,
                                                           out_shape=(out_img.shape[1], out_img.shape[2]))

            convex_mask_tif = None
            if use_convex_mask and crowns is not None:
                # Create a convex mask around the crown polygons
                if gpd.__version__.startswith("1."):
                    unioned_crowns = overlapping_crowns.union_all()
                else:
                    unioned_crowns = overlapping_crowns.unary_union
                convex_mask_tif = rasterio.features.geometry_mask([unioned_crowns.convex_hull.buffer(5)],
                                                                  transform=out_transform,
                                                                  invert=True,
                                                                  out_shape=(out_img.shape[1], out_img.shape[2]))

            out_sumbands = np.sum(out_img, axis=0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.isnan(out_sumbands) | np.where(out_sumbands == 765, 1, 0) | np.where(
                out_sumbands == 65535 * 3, 1, 0)
            for nodata_val in additional_nodata:
                nan_mask = nan_mask | np.where(out_sumbands == nodata_val, 1, 0)
            if mask_tif is not None:
                nan_mask = nan_mask | ~mask_tif
            if convex_mask_tif is not None:
                nan_mask = nan_mask | ~convex_mask_tif
            totalpix = out_img.shape[1] * out_img.shape[2]

            # If the tile is mostly empty or mostly nan, don't save it
            invalid = (zero_mask | nan_mask).sum()
            if invalid > nan_threshold * totalpix:
                logger.warning(
                    f"Skipping tile at ({minx}, {miny}) due to being over nodata threshold. Threshold: {nan_threshold}, nodata ration: {invalid / totalpix}"
                )
                return None

            # Apply nan mask
            out_img[np.broadcast_to((nan_mask == 1)[None, :, :], out_img.shape)] = 0

            dtype, nodata = dtype_map.get(out_img.dtype, (None, None))
            if dtype is None:
                logger.exception(f"Unsupported dtype: {out_img.dtype}")

            out_meta = data.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "count": 3,
                "nodata": nodata,
                "dtype": dtype,
            })

            out_tif = out_path_root.with_suffix(".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            r, g, b = out_img[0], out_img[1], out_img[2]
            rgb = np.dstack((b, g, r))  # Reorder for cv2 (BGRA)

            # Rescale to 0-255 if necessary
            if np.nanmax(g) > 255:
                rgb_rescaled = rgb / 65535 * 255
            else:
                rgb_rescaled = rgb

            np.clip(rgb_rescaled, 0, 255, out=rgb_rescaled)

            cv2.imwrite(str(out_path_root.with_suffix(".png").resolve()), rgb_rescaled.astype(np.uint8))

            if overlapping_crowns is not None:
                return data, out_path_root, overlapping_crowns, minx, miny, buffer

            return data, out_path_root, None, minx, miny, buffer

    except RasterioIOError as e:
        logger.error(f"RasterioIOError while applying mask {coords}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing tile {tilename} at ({minx}, {miny}): {e}")
        return None


def process_tile_ms(img_path: str,
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
                    mask_gdf: gpd.GeoDataFrame = None,
                    additional_nodata: List[Any] = [],
                    image_statistics: List[Dict[str, float]] = None,
                    ignore_bands_indices: List[int] = [],
                    use_convex_mask: bool = True):
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
                if mask_gdf is not None:
                    overlapping_crowns = gpd.clip(overlapping_crowns, mask_gdf)
                if overlapping_crowns.empty or (overlapping_crowns.dissolve().area[0] / geo.area[0]) < threshold:
                    return None
            else:
                overlapping_crowns = None
            if data.nodata is not None:
                nodata = data.nodata
            else:
                nodata = 0

            bands_to_read = [
                i for i in list(range(1, data.count + 1)) if i not in [i + 1 for i in ignore_bands_indices]
            ]
            out_img, out_transform = mask(data, shapes=coords, nodata=nodata, crop=True, indexes=bands_to_read)

            mask_tif = None
            if mask_gdf is not None:
                #if mask_gdf.crs != data.crs:
                #    mask_gdf = mask_gdf.to_crs(data.crs) #TODO is this necessary?

                mask_tif = rasterio.features.geometry_mask([geom for geom in mask_gdf.geometry],
                                                           transform=out_transform,
                                                           invert=True,
                                                           out_shape=(out_img.shape[1], out_img.shape[2]))

            convex_mask_tif = None
            if use_convex_mask and crowns is not None:
                # Create a convex mask around the crown polygons
                if gpd.__version__.startswith("1."):
                    unioned_crowns = overlapping_crowns.union_all()
                else:
                    unioned_crowns = overlapping_crowns.unary_union
                convex_mask_tif = rasterio.features.geometry_mask([unioned_crowns.convex_hull.buffer(5)],
                                                                  transform=out_transform,
                                                                  invert=True,
                                                                  out_shape=(out_img.shape[1], out_img.shape[2]))

            out_sumbands = np.sum(out_img, axis=0)
            zero_mask = np.where(out_sumbands == 0, 1, 0)
            nan_mask = np.isnan(out_sumbands) | np.where(out_sumbands == 65535 * len(bands_to_read), 1, 0)
            for nodata_val in additional_nodata:
                nan_mask = nan_mask | np.where(out_sumbands == nodata_val, 1, 0)
            if mask_tif is not None:
                nan_mask = nan_mask | ~mask_tif
            if convex_mask_tif is not None:
                nan_mask = nan_mask | ~convex_mask_tif
            totalpix = out_img.shape[1] * out_img.shape[2]

            # If the tile is mostly empty or mostly nan, don't save it
            invalid = (zero_mask | nan_mask).sum()
            if invalid > nan_threshold * totalpix:
                logger.warning(
                    f"Skipping tile at ({minx}, {miny}) due to being over nodata threshold. Threshold: {nan_threshold}, nodata ration: {invalid / totalpix}"
                )
                return None

            # rescale image to 1-255 (0 is reserved for nodata)
            assert image_statistics is not None, "image_statistics must be provided for multispectral data"
            min_vals = np.array([stats['min'] for stats in image_statistics]).reshape(-1, 1, 1)
            max_vals = np.array([stats['max'] for stats in image_statistics]).reshape(-1, 1, 1)

            # making it a bit safer for small numbers
            if max_vals.min() > 1:
                out_img = (out_img - min_vals) / (max_vals - min_vals) * 254 + 1
            else:
                out_img = (out_img - min_vals) * 254 / (max_vals - min_vals) + 1

            # additional clip to make sure
            out_img = np.clip(out_img.astype(np.float32), 1.0, 255.0)

            # Apply nan mask
            out_img[np.broadcast_to((nan_mask == 1)[None, :, :], out_img.shape)] = 0.0

            dtype, nodata = dtype_map.get(out_img.dtype, (None, None))
            if dtype is None:
                logger.exception(f"Unsupported dtype: {out_img.dtype}")

            out_meta = data.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_img.shape[1],
                "width": out_img.shape[2],
                "transform": out_transform,
                "count": out_img.shape[0],
                "nodata": nodata,
                "dtype": dtype
            })

            out_tif = out_path_root.with_suffix(".tif")
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            # Uncomment, if pngs of the first three channels of the ms image are needed
            # r, g, b = out_img[0], out_img[1], out_img[2]
            # rgb = np.dstack((b, g, r)).astype(np.uint8)  # Reorder for cv2 (BGRA)
            # cv2.imwrite(str(out_path_root.with_suffix(".png").resolve()), rgb)

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
        mask_gdf: gpd.GeoDataFrame = None,
        additional_nodata: List[Any] = [],
        image_statistics: List[Dict[str, float]] = None,
        ignore_bands_indices: List[int] = [],
        use_convex_mask: bool = True) -> None:
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
        result = process_tile(img_path, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs, tilename,
                              crowns, threshold, nan_threshold, mask_gdf, additional_nodata, image_statistics,
                              ignore_bands_indices, use_convex_mask)
    elif mode == "ms":
        result = process_tile_ms(img_path, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs,
                                 tilename, crowns, threshold, nan_threshold, mask_gdf, additional_nodata,
                                 image_statistics, ignore_bands_indices, use_convex_mask)

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


def _calculate_tile_placements(
    img_path: str,
    buffer: int,
    tile_width: int,
    tile_height: int,
    crowns: gpd.GeoDataFrame = None,
    tile_placement: str = "grid",
    overlapping_tiles: bool = False,
) -> List[Tuple[int, int]]:
    """Internal method for calculating the placement of tiles"""

    if tile_placement == "grid":
        with rasterio.open(img_path) as data:
            coordinates = [
                (minx, miny) for minx in np.arange(
                    math.ceil(data.bounds[0]) + buffer, data.bounds[2] - tile_width - buffer, tile_width, int)
                for miny in np.arange(
                    math.ceil(data.bounds[1]) + buffer, data.bounds[3] - tile_height - buffer, tile_height, int)
            ]
            if overlapping_tiles:
                coordinates.extend([(minx, miny) for minx in np.arange(
                    math.ceil(data.bounds[0]) + buffer + tile_width // 2, data.bounds[2] - tile_width - buffer -
                    tile_width // 2, tile_width, int) for miny in np.arange(
                        math.ceil(data.bounds[1]) + buffer + tile_height // 2, data.bounds[3] - tile_height - buffer -
                        tile_height // 2, tile_height, int)])
    elif tile_placement == "adaptive":

        if crowns is None:
            logger.warning(
                'Crowns must be supplied if tile_placement="adaptive" (crowns is None). Assuming tiling for test dataset, and tile placement will be done with tile_placement == "grid" instead.'
            )
            return _calculate_tile_placements(img_path, buffer, tile_width, tile_height)

        logger.info("Starting Union of Crowns")
        if gpd.__version__.startswith("1."):
            unioned_crowns = crowns.union_all()
        else:
            unioned_crowns = crowns.unary_union
        logger.info(f"Finished Union of Crowns")

        area_width = crowns.total_bounds[2] - crowns.total_bounds[0]
        area_height = crowns.total_bounds[3] - crowns.total_bounds[1]
        required_tiles_x = math.ceil(area_width / tile_width)
        required_tiles_y = math.ceil(area_height / tile_height)
        combined_tiles_width = required_tiles_x * tile_width
        combined_tiles_height = required_tiles_y * tile_height
        x_offset = (combined_tiles_width - area_width) / 2
        y_offset = (combined_tiles_height - area_height) / 2

        logger.info("Starting Tile Placement Generation")
        coordinates = []
        for row in range(required_tiles_y):
            bar = gpd.GeoSeries([
                box(crowns.total_bounds[0] - x_offset, crowns.total_bounds[1] - y_offset + row * tile_height,
                    crowns.total_bounds[2] + x_offset, crowns.total_bounds[1] - y_offset + (row + 1) * tile_height)
            ],
                                crs=crowns.crs)

            intersection = unioned_crowns.intersection(bar)
            if intersection.is_empty.all():
                continue

            intersection_width = intersection.total_bounds[2] - intersection.total_bounds[0]
            required_intersection_tiles_x = math.ceil(intersection_width / tile_width)
            combined_intersection_tiles_width = required_intersection_tiles_x * tile_width
            x_intersection_offset = (combined_intersection_tiles_width - intersection_width) / 2

            for col in range(required_intersection_tiles_x):
                coordinates.append((int(intersection.total_bounds[0] - x_intersection_offset) + col * tile_width,
                                    int(crowns.total_bounds[1] - y_offset) + row * tile_height))
                if overlapping_tiles:
                    coordinates.append(
                        (int(intersection.total_bounds[0] - x_intersection_offset) + col * tile_width + tile_width // 2,
                         int(crowns.total_bounds[1] - y_offset) + row * tile_height + tile_height // 2))
        logger.info(f"Finished Tile Placement Generation")
    else:
        raise ValueError('Unsupported tile_placement method. Must be "grid" or "adaptive"')

    return coordinates


def calculate_image_statistics(file_path,
                               values_to_ignore=None,
                               window_size=64,
                               min_windows=100,
                               mode="rgb",
                               ignore_bands_indices: List[int] = []):
    """
    Calculate statistics for a raster using either whole image or sampled windows.

    Parameters:
    - file_path: str, path to the raster file.
    - values_to_ignore: list, values to ignore in statistics (e.g., NaN, custom values).
    - window_size: int, size of square window for sampling.
    - min_windows: int, minimum number of valid windows to include in statistics.

    Returns:
    - List of dictionaries containing statistics for each band.
    """
    if values_to_ignore is None:
        values_to_ignore = []
    values_to_ignore.append(65535)
    with rasterio.open(file_path) as src:
        # Get image dimensions
        width, height = src.width, src.height

        def calc_on_everything():
            logger.info("Processing entire image...")
            band_stats = []
            for band_idx in range(1, src.count + 1):
                if band_idx - 1 in ignore_bands_indices:
                    continue
                band = src.read(band_idx).astype(float)
                # Mask out bad values
                mask = (np.isnan(band) | np.isin(band, values_to_ignore))
                valid_data = band[~mask]

                if valid_data.size > 0:
                    min_val, max_val = np.percentile(valid_data, [1, 99])

                    stats = {
                        "mean": np.mean(valid_data),
                        "min": min_val,
                        "max": max_val,
                        "std_dev": np.std(valid_data),
                    }
                else:
                    stats = {
                        "mean": None,
                        "min": None,
                        "max": None,
                        "std_dev": None,
                    }
                band_stats.append(stats)
            return band_stats

        # If the image is smaller than 2000x2000, process the whole image
        if width * height <= 2000 * 2000:
            return calc_on_everything()

        windows_sampled = 0
        band_aggregates: Dict[int, List[Union[float, int]]] = {}
        i = 1
        for band_idx in range(1, src.count + 1):
            if band_idx - 1 in ignore_bands_indices:
                continue
            band_aggregates[i] = []
            i += 1

        fails = 0
        while windows_sampled < min_windows:
            if fails > 100:
                logger.warning(
                    "Too many failed windows for caluclating image statistics. Calculating on whole image instead.")
                return calc_on_everything()
            # Randomly pick a top-left corner for the window
            row_start = np.random.randint(0, height - window_size)
            col_start = np.random.randint(0, width - window_size)

            window = rasterio.windows.Window(col_start, row_start, window_size, window_size)

            # Read the window for each band
            valid_window = True
            window_data = {}
            i = 1
            for band_idx in range(1, src.count + 1) if mode == "ms" else range(1, 4):
                if band_idx - 1 in ignore_bands_indices:
                    continue
                band = src.read(band_idx, window=window).astype(float)
                # Mask out bad values
                mask = (np.isnan(band) | np.isin(band, values_to_ignore))
                valid_pixels = band[~mask]
                bad_pixel_ratio = mask.sum() / band.size

                if bad_pixel_ratio > 0.05:  # Exclude windows with >5% bad values
                    valid_window = False
                    logger.info(
                        f"Window {windows_sampled} has too many bad pixels ({bad_pixel_ratio:.2f}). Skipping...")
                    fails += 1
                    break
                window_data[i] = valid_pixels
                i += 1

            if valid_window:
                for band_idx, valid_pixels in window_data.items():
                    band_aggregates[band_idx].extend(valid_pixels)
                windows_sampled += 1

        # Compute statistics for each band
        band_stats = []
        for band_idx in range(1, len(band_aggregates) + 1) if mode == "ms" else range(1, 4):
            valid_data = np.array(band_aggregates[band_idx])
            if valid_data.size > 0:
                min_val, max_val = np.percentile(valid_data, [1, 99])
                stats = {
                    "mean": np.mean(valid_data),
                    "min": min_val,
                    "max": max_val,
                    "std_dev": np.std(valid_data),
                }
            else:
                stats = {
                    "mean": None,
                    "min": None,
                    "max": None,
                    "std_dev": None,
                }
            band_stats.append(stats)
        return band_stats


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
    tile_placement: str = "grid",
    mask_path: str = None,
    multithreaded: bool = False,
    random_subset: int = -1,
    additional_nodata: List[Any] = [],
    overlapping_tiles: bool = False,
    ignore_bands_indices: List[int] = [],
    use_convex_mask: bool = True,
) -> None:
    """Tiles up orthomosaic and corresponding crowns (if supplied) into training/prediction tiles.

    Tiles up large rasters into manageable tiles for training and prediction. If crowns are not supplied, the function
    will tile up the entire landscape for prediction. If crowns are supplied, the function will tile these with the image
    and skip tiles without a minimum coverage of crowns. The 'threshold' can be varied to ensure good coverage of
    crowns across a training tile. Tiles that do not have sufficient coverage are skipped.

    Args:
        img_path: Path to the orthomosaic
        out_dir: Output directory
        buffer: Overlapping buffer of tiles in meters (UTM)
        tile_width: Tile width in meters
        tile_height: Tile height in meters
        crowns: Crown polygons as a GeoPandas DataFrame
        threshold: Minimum proportion of the tile covered by crowns to be accepted [0,1]
        nan_threshold: Maximum proportion of tile covered by NaNs [0,1]
        dtype_bool: Flag to edit dtype to prevent black tiles
        mode: Type of the raster data ("rgb" or "ms")
        class_column: Name of the column in `crowns` DataFrame for class-based tiling
        tile_placement: Strategy for placing tiles.
            "grid" for fixed grid placement based on the bounds of the input image, optimized for speed.
            "adaptive" for dynamic placement of tiles based on crowns, adjusts based on data features for better coverage.

    Returns:
        None
    """

    mask_gdf: gpd.GeoDataFrame = None
    if mask_path is not None:
        mask_gdf = gpd.read_file(mask_path)
    out_path = Path(out_dir)
    os.makedirs(out_path, exist_ok=True)
    tilename = Path(img_path).stem
    with rasterio.open(img_path) as data:
        crs = data.crs.to_epsg()  # Update CRS handling to avoid deprecated syntax

    tile_coordinates = _calculate_tile_placements(img_path, buffer, tile_width, tile_height, crowns, tile_placement,
                                                  overlapping_tiles)
    image_statistics = calculate_image_statistics(img_path,
                                                  values_to_ignore=additional_nodata,
                                                  mode=mode,
                                                  ignore_bands_indices=ignore_bands_indices)

    tile_args = [
        (img_path, out_dir, buffer, tile_width, tile_height, dtype_bool, minx, miny, crs, tilename, crowns, threshold,
         nan_threshold, mode, class_column, mask_gdf, additional_nodata, image_statistics, ignore_bands_indices,
         use_convex_mask) for minx, miny in tile_coordinates
        if mask_path is None or (mask_path is not None and mask_gdf.intersects(
            box(minx, miny, minx + tile_width, miny + tile_height)  #TODO maybe add to_crs here
        ).any())
    ]

    if random_subset > -1:
        if random_subset > len(tile_args):
            logger.warning(
                f"random_subset is larger than the amount of tile places ({len(tile_args)}>{random_subset}). Using all possible tiles instead."
            )
        else:
            tile_args = random.sample(tile_args, random_subset)

    if multithreaded:
        total_tiles = len(tile_args)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_tile_train_helper, arg) for arg in tile_args]
            with tqdm(total=total_tiles) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.exception(f'Tile generated an exception: {exc}')
                    pbar.update(1)
    else:
        for args in tqdm(tile_args):
            process_tile_train_helper(args)

    logger.info("Tiling complete")


def create_RGB_from_MS(tile_folder_path: Union[str, Path],
                       out_dir: Union[str, Path, None] = None,
                       conversion: str = "pca",
                       memory_limit: float = 3.0) -> None:
    """
    Convert a collection of multispectral GeoTIFF tiles into three-band RGB images. By default, uses PCA
    to reduce all bands to three principal components, or extracts the first three bands if 'first_three'
    is chosen. The function saves each output as both a three-band GeoTIFF and a PNG preview.

    Args:
        tile_folder_path (str or Path):
            Path to the folder containing multispectral .tif files, along with any .geojson, train, or test subdirectories.
        out_dir (str or Path, optional):
            Path to the output directory where RGB images will be saved. If None, a default folder with a suffix
            "_<conversion>-rgb" is created alongside the input tile folder. If `out_dir` already exists and is not empty,
            we append also append the current date and time to avoid overwriting.
        conversion (str, optional):
            The method of converting multispectral imagery to three bands:
            - "pca": perform a principal-component analysis reduction to three components.
            - "first-three": simply select the first three bands (i.e., B1 -> R, B2 -> G, B3 -> B).
        memory_limit (float, optional):
            Approximate memory limit in gigabytes for loading data to compute PCA.
            Relevant only if `conversion="pca"`.

    Returns:
        None

    Raises:
        ValueError:
            If no data is found in the tile folder or if an unsupported conversion method is specified.
        ImportError:
            If scikit-learn is not installed and `conversion="pca"`.
        RasterioIOError:
            If reading any of the .tif files fails.
    """

    # Ensure we are working with Path objects
    tile_folder = Path(tile_folder_path).resolve()

    # Determine the default or custom output folder
    if out_dir is None:
        # Example: input folder is /home/user/tiles -> out_dir = /home/user/tiles_pca-rgb
        out_path = tile_folder.parent / f"{tile_folder.name}_{conversion}-rgb"
    else:
        out_path = Path(out_dir).resolve()
        # If user-specified out_dir is non-empty, append the tile_folder name to avoid overwriting
        if out_path.is_dir() and any(out_path.iterdir()):
            out_path = out_path / f"{tile_folder.name}_{conversion}-rgb"

    # If the resolved output folder already exists, append a timestamp to avoid overwriting
    if out_path.exists():
        out_path = Path(str(out_path) + f"-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    out_path.mkdir(parents=True, exist_ok=False)

    # Gather a list of all .tif files and shuffle them
    tif_files = list(tile_folder.glob("*.tif"))
    random.shuffle(tif_files)

    if not tif_files:
        raise ValueError(f"No .tif files found in {tile_folder}")

    # ----------------------------------------------------------------------------------------
    # If conversion == "pca", gather data from the tiles up to the memory limit, compute PCA.
    # ----------------------------------------------------------------------------------------
    if conversion == "pca":
        images = []
        size_in_gb = 0.0

        # 1. Collect data (up to memory_limit GB) for PCA
        for tif_file in tqdm(tif_files, desc="Collecting data for PCA"):
            try:
                with rasterio.open(tif_file) as src:
                    data = src.read().reshape(src.count, -1).T  # Shape: (pixels, bands)
            except RasterioIOError as e:
                logger.error(f"Failed to read file {tif_file}: {e}")
                continue

            # Exclude all-zero (nodata) rows to reduce memory usage
            data_sum = np.sum(data, axis=1)
            non_zero_mask = data_sum != 0.0
            filtered_data = data[non_zero_mask]
            images.append(filtered_data)

            # Track approximate memory usage
            size_in_gb += filtered_data.nbytes / (1024**3)

            # If we exceed the memory limit, stop collecting
            if size_in_gb > memory_limit / 2:
                logger.info(f"Memory limit reached at ~{len(images)} chunks. Proceeding with PCA calculation.")
                break

        # Make sure we have some data to fit PCA
        if not images:
            raise ValueError("No valid data found in the folder for PCA.")

        # 2. Concatenate all non-zero pixel data for PCA
        images_concatenated = np.concatenate(images, axis=0)
        del images  # Clean up memory

        # 3. Fit incremental PCA
        try:
            from sklearn.decomposition import IncrementalPCA
        except ImportError:
            logger.error("scikit-learn is required for PCA conversion. Please install scikit-learn to continue.")
            raise ImportError("scikit-learn not installed.")

        batch_size = 1_000_000
        logger.info("Incrementally calculating principal components...")
        ipca = IncrementalPCA(n_components=3, batch_size=batch_size)
        ipca_data = ipca.fit_transform(images_concatenated)
        del images_concatenated  # Freed up once PCA is fit

        # 4. Compute a robust range (1-99 percentile) to map PCA outputs to [1,255]
        min_vals, max_vals = np.percentile(ipca_data, [1, 99], axis=0)
        logger.info(f"PCA explained variance ratio: {ipca.explained_variance_ratio_}")
        del ipca_data  # Freed up once min/max are computed

        # 5. Transform each tile and save as .tif and .png
        for tif_file in tqdm(tif_files, desc="Applying PCA and saving RGB"):
            try:
                with rasterio.open(tif_file) as src:
                    transform = src.transform
                    crs = src.crs
                    out_meta = src.meta.copy()
                    raw_data = src.read().reshape(src.count, -1).T  # (pixels, bands)
            except RasterioIOError as e:
                logger.error(f"Failed to read file {tif_file}: {e}")
                continue

            width, height = src.width, src.height
            data_sum = np.sum(raw_data, axis=1)
            zero_mask = (data_sum == 0.0)

            # Project the tile's data into the PCA components
            transformed = ipca.transform(raw_data).astype(np.float32)
            # Rescale to [1,255]
            transformed = (transformed - min_vals) / (max_vals - min_vals) * 254.0 + 1.0
            transformed = np.clip(transformed, 1.0, 255.0)

            # Reinstate nodata=0 for zero rows
            transformed[zero_mask] = 0.0

            # Reshape back into (bands=3, height, width)
            transformed = transformed.T.reshape(3, height, width)

            # Optional reorder if you want IPCA dimension #1 as R, #2 as G, #3 as B.
            # The snippet below swaps them to [G, R, B], but feel free to remove or alter.
            transformed = transformed[[1, 0, 2], :, :]

            # Update meta for writing out
            out_meta.update({
                "nodata": 0.0,
                "dtype": "float32",
                "count": 3,
                "width": width,
                "height": height,
                "transform": transform,
                "crs": crs
            })

            # Write the GeoTIFF
            output_tif = out_path / tif_file.name
            with rasterio.open(output_tif, "w", **out_meta) as dest:
                dest.write(transformed)

            # Write the PNG (we must convert shape to (H, W, 3) and then to uint8)
            output_png = out_path / f"{tif_file.stem}.png"
            png_ready = np.moveaxis(transformed, 0, -1).astype(np.uint8)  # (H, W, 3)
            cv2.imwrite(str(output_png), cv2.cvtColor(png_ready, cv2.COLOR_RGB2BGR))

    elif conversion == "first-three":
        # ----------------------------------------------------------------------------------------
        # If conversion == "first_three", simply take the first 3 bands and save them as RGB.
        # ----------------------------------------------------------------------------------------
        for tif_file in tqdm(tif_files, desc="Selecting first three bands"):
            try:
                with rasterio.open(tif_file) as src:
                    out_meta = src.meta.copy()
                    # Read only the first three bands (or fewer if the file has < 3 bands)
                    data = src.read(indexes=[1, 2, 3])
                    if np.nanmax(data) > 255:
                        logger.exception(
                            "The input folder seems to be an RGB folder and you are taking the first three bands. This will not change the output. Did you choose the wrong folder? Aborting."
                        )
                        return
            except RasterioIOError as e:
                logger.error(f"Failed to read file {tif_file}: {e}")
                continue

            # If the file has fewer than 3 bands, we can pad the missing bands with zeros
            if data.shape[0] < 3:
                raise ValueError(f"File {tif_file} has fewer than 3 bands.")

            # Update meta to reflect 3 bands, 8-bit
            out_meta.update({
                "count": 3,
            })

            # Write out the new GeoTIFF
            output_tif = out_path / tif_file.name
            with rasterio.open(output_tif, 'w', **out_meta) as dest:
                dest.write(data)

            # Write out the PNG (shape must be (H, W, 3))
            output_png = out_path / f"{tif_file.stem}.png"
            # Move axis from (bands, H, W) -> (H, W, bands)
            png_ready = np.moveaxis(data, 0, -1).astype(np.uint8)
            # We expect the order to be [band1, band2, band3], so interpret as R,G,B
            cv2.imwrite(str(output_png), cv2.cvtColor(png_ready, cv2.COLOR_RGB2BGR))

    else:
        raise ValueError(f"Unsupported conversion method: {conversion}")

    # ------------------------------------------------------------------------------
    # Copy .geojson files and train/test folders from the source to the out folder
    # ------------------------------------------------------------------------------
    for geojson_file in tile_folder.glob("*.geojson"):
        shutil.copy(str(geojson_file), str(out_path / geojson_file.name))

    # If a 'train' folder exists in the source, copy it to the out_path
    train_subfolder = tile_folder / "train"
    if train_subfolder.is_dir():
        shutil.copytree(str(train_subfolder), str(out_path / "train"))

    # If a 'test' folder exists in the source, copy it to the out_path
    test_subfolder = tile_folder / "test"
    if test_subfolder.is_dir():
        shutil.copytree(str(test_subfolder), str(out_path / "test"))

    # ------------------------------------------------------------------------
    # Update "imagePath" fields in .geojson to point to new .png (instead of .tif)
    # ------------------------------------------------------------------------
    # 1. Collect geojson files directly under out_path
    direct_geojsons = list(out_path.glob("*.geojson"))

    # 2. Collect geojson files in test subfolder
    files_in_subdirs = []
    test_folder = out_path / "test"
    if test_folder.is_dir():
        files_in_subdirs.extend(list(test_folder.glob("*.geojson")))

    # 3. Collect geojson files in train/fold_* subfolders
    train_folder = out_path / "train"
    if train_folder.is_dir():
        fold_dirs = list(train_folder.glob("fold_*"))
        for fold_dir in fold_dirs:
            if fold_dir.is_dir():
                files_in_subdirs.extend(list(fold_dir.glob("*.geojson")))

    # Merge all .geojson paths for final processing
    all_geojsons = direct_geojsons + files_in_subdirs

    pattern = r'"imagePath":\s*"([^"]+)"'
    for file_path in tqdm(all_geojsons, desc="Updating 'imagePath' in GeoJSONs"):
        file_path = file_path.resolve()
        with open(file_path, 'r') as f:
            content = f.read()

        match = re.search(pattern, content)
        if match:
            old_image_path = match.group(1)
            old_image_filename = Path(old_image_path).name
            # Replace '.tif' with '.png' if it exists in the name
            new_image_filename = old_image_filename.replace('.tif', '.png')
            new_image_path = (out_path / new_image_filename).as_posix()

            # Update the content with the new path
            new_content = re.sub(pattern, f'"imagePath": "{new_image_path}"', content)

            # Overwrite the geojson file with the updated image path
            with open(file_path, 'w') as f:
                f.write(new_content)
        else:
            logger.info(f"No 'imagePath' found in {file_path.name}, skipping.")

    # -------------------------------------------------------------------------
    # Optional verification step: check that .geojson files in subfolders match
    # any corresponding file in the main directory
    # -------------------------------------------------------------------------
    for sub_file in tqdm(files_in_subdirs, desc="Verifying .geojson files in subfolders"):
        sub_file_name = sub_file.name
        # Construct the path it might have in the main out_path
        expected_file = out_path / sub_file_name
        if not expected_file.is_file():
            logger.warning(f"Missing .geojson file in main out_dir: {expected_file}")
            continue

        sub_size = os.path.getsize(str(sub_file))
        main_size = os.path.getsize(str(expected_file))
        if sub_size != main_size:
            logger.warning(f"Size mismatch for {sub_file_name}:")
            logger.warning(f" - Size in subdir: {sub_size} bytes")
            logger.warning(f" - Size in out_dir: {main_size} bytes")

    # Copy anything that is not a .tif, .png or .geojson file to the output folder (ignore folders)
    for file in tile_folder.glob("*"):
        if file.is_file() and file.suffix not in [".tif", ".png", ".geojson"]:
            shutil.copy(str(file), str(out_path))

    logger.info("RGB creation from multispectral complete.")


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

    logger.info(f"Classes saved as {save_format} file: {class_to_idx}")


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
            shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(Path(file_roots[num[i]]).suffix + ".geojson"),
                        out_dir / "test")
        else:
            # copy to train
            train_box = image_details(file_roots[num[i]])
            if strict:  # check if there is overlap with test boxes
                if not is_overlapping_box(test_boxes, train_box):
                    shutil.copy(
                        (tiles_dir / file_roots[num[i]]).with_suffix(Path(file_roots[num[i]]).suffix + ".geojson"),
                        out_dir / "train")
            else:
                shutil.copy((tiles_dir / file_roots[num[i]]).with_suffix(Path(file_roots[num[i]]).suffix + ".geojson"),
                            out_dir / "train")

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
    mask_path = "/media/chris/wd-elements1/data/Harapan_2019/mask_clipped.gpkg"

    # Optional parameters for tiling and processing
    buffer = 30  # Overlap between tiles (in meters)
    tile_width = 200  # Tile width (in meters)
    tile_height = 200  # Tile height (in meters)
    nan_threshold = 0.1  # Max proportion of tile that can be NaN before it's discarded
    threshold = 0.5  # Minimum crown coverage per tile for it to be kept (0-1)
    dtype_bool = False  # Change dtype to uint8 to avoid black tiles
    mode = "rgb"  # Use 'rgb' for regular 3-channel imagery, 'ms' for multispectral
    class_column = "species"  # Column in the crowns file to use as the class label
    tile_placement = "adaptive"  # Determines the way that tiles are are placed, can be either "grid" or "adaptive"

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
        class_column=class_column,  # Use the selected class column (e.g., 'species', 'status')
        tile_placement=tile_placement,
        mask_path=mask_path)

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
