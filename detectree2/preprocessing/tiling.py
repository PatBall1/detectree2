# necessary basic libraries
import numpy as np
import cv2
import json
import rasterio
import geopandas
from geopandas.tools import sjoin
from rasterio.mask import mask
from rasterio.io import DatasetReader
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg


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


def getFeatures(gdf):
    """
    Function to parse features from GeoDataFrame in such a manner that rasterio wants them
    """
    return [json.loads(gdf.to_json())["features"][0]["geometry"]]


def tile_data(data, out_dir, buffer=30, tile_width=200, tile_height=200, crowns=None):
    """
    Function to tile up image and (if included) corresponding crowns
    """
    for minx in np.arange(data.bounds[0], data.bounds[2] - tile_width, tile_width, int):
        # print("minx:", minx)
        for miny in np.arange(
            data.bounds[1], data.bounds[3] - tile_height, tile_height, int
        ):
            # print("miny:", miny)

            # new tiling bbox including the buffer
            bbox = box(
                minx - buffer,
                miny - buffer,
                minx + tile_width + buffer,
                miny + tile_height + buffer,
            )
            # define the bounding box of the tile, excluding the buffer (hence selecting just the central part of the tile)
            bbox_central = box(minx, miny, minx + tile_width, miny + tile_height)

            # turn the bounding boxes into geopandas DataFrames
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=from_epsg(4326))
            geo_central = gpd.GeoDataFrame(
                {"geometry": bbox_central}, index=[0], crs=from_epsg(4326)
            )  # 3182

            # here we are cropping the tiff to the bounding box of the tile we want
            coords = getFeatures(geo)
            # print("Coords:", coords)

            # define the tile as a mask of the whole tiff with just the bounding box
            out_img, out_transform = mask(data, shapes=coords, crop=True)
            # print('out transform:', out_transform)

            # This can be useful when reprojecting later as know the crs format to put it into
            # epsg_code = int(data.crs.data["init"][5:])
            # print(epsg_code)

            # copy the metadata then update it, the "nodata" and "dtype" where important as made larger
            # tifs have outputted tiles which were not just black
            out_meta = data.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "nodata": None,
                    "dtype": "uint8",
                }
            )
            # print('Out Meta:',out_meta)

            # Saving the tile as a new tiff, named by the origin of the tile. If tile appears blank in folder can show the image here and may
            # need to fix RGB data or the dtype
            # show(out_img)
            out_tif = out_dir + "tile_" + str(minx) + "_" + str(miny) + ".tif"
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            # read in the tile we have just saved
            clipped = rasterio.open(
                out_dir + "/tile_" + str(minx) + "_" + str(miny) + ".tif"
            )
            # read it as an array
            # show(clipped)
            arr = clipped.read()

            # each band of the tiled tiff is a colour!
            R = arr[0]
            G = arr[1]
            B = arr[2]

            # stack up the bands in an order appropriate for saving with cv2, then rescale to the correct 0-255 range for cv2

            rgb = np.dstack((B, G, R))  # BGR for cv2
            rgb_rescaled = rgb  # scale to image
            # print('rgb rescaled', rgb_rescaled)

            # save this as jpg or png...we are going for png...again, named with the origin of the specific tile
            # here as a naughty method
            cv2.imwrite(
                out_dir + "tile_" + str(minx) + "_" + str(miny) + ".png",
                rgb_rescaled,
            )

            # img = cv2.imread(
            #    "gdrive/MyDrive/JamesHirst/NY/LargeArea_images/naip_cayuga/naip_cayuga_tiled_by_me/tile_"
            #    + str(minx)
            #    + "_"
            #    + str(miny)
            #    + ".png"
            # )
            # print('png shape:', img.shape)
            if crowns is not None:
                # select the crowns that intersect the non-buffered central
                # section of the tile using the inner join
                # JB : a better solution would be to clip crowns to tile extent
                overlapping_crowns = sjoin(crowns, geo_central, how="inner")
                overlapping_crowns = overlapping_crowns.explode(index_parts=True)
                # print("Overlapping crowns:", overlapping_crowns)

                # translate to 0,0 to overlay on png
                # this now works as a universal approach.
                if minx == data.bounds[0] and miny == data.bounds[1]:
                    # print("We are in the bottom left!")
                    moved = overlapping_crowns.translate(-minx, -miny)
                elif miny == data.bounds[1]:
                    # print("We are on the bottom, but not bottom left")
                    moved = overlapping_crowns.translate(-minx + buffer, -miny)
                elif minx == data.bounds[0]:
                    # print("We are along the left hand side, but not bottom left!")
                    moved = overlapping_crowns.translate(-minx, -miny + buffer)
                else:
                    # print("We are in the middle!")
                    moved = overlapping_crowns.translate(-minx + buffer, -miny + buffer)
                # print("Moved coords:", moved)

                # scale to deal with the resolution
                scalingx = 1 / (data.transform[0])
                scalingy = -1 / (data.transform[4])
                moved_scaled = moved.scale(scalingx, scalingy, origin=(0, 0))
                # print(moved_scaled)
                impath = {
                    "imagePath": (
                        out_dir + "tile_" + str(minx) + "_" + str(miny) + ".png"
                    )
                }

                # save as a geojson, a format compatible with detectron2, again named by the origin of the tile.
                # If the box selected from the image is outside of the mapped region due to the image being on a slant
                # then the shp file will have no info on the crowns and hence will create an empty gpd Dataframe.
                # this causes an error so skip creating geojson. The training code will also ignore png so no problem.
                try:
                    filename = "./tile_" + str(minx) + "_" + str(miny) + ".geojson"
                    moved_scaled.to_file(
                        driver="GeoJSON",
                        filename=filename,
                    )
                    with open(filename, "r") as f:
                        shp = json.load(f)
                        shp.update(impath)
                    with open(filename, "w") as f:
                        json.dump(shp, f)
                except:
                    print("ValueError: Cannot write empty DataFrame to file.")
                    continue


def tile_data_reduced(
    data, out_dir, buffer=30, tile_width=200, tile_height=200, crowns=None, threshold=0
):
    """
    Function to tile up image and (if included) corresponding crowns.
    Only outputs tiles with crowns in.
    """
    for minx in np.arange(data.bounds[0], data.bounds[2] - tile_width, tile_width, int):
        # print("minx:", minx)
        for miny in np.arange(
            data.bounds[1], data.bounds[3] - tile_height, tile_height, int
        ):
            # print("miny:", miny)

            # new tiling bbox including the buffer
            bbox = box(
                minx - buffer,
                miny - buffer,
                minx + tile_width + buffer,
                miny + tile_height + buffer,
            )
            # define the bounding box of the tile, excluding the buffer (hence selecting just the central part of the tile)
            bbox_central = box(minx, miny, minx + tile_width, miny + tile_height)

            # turn the bounding boxes into geopandas DataFrames
            geo = gpd.GeoDataFrame({"geometry": bbox}, index=[0], crs=from_epsg(4326))
            geo_central = gpd.GeoDataFrame(
                {"geometry": bbox_central}, index=[0], crs=from_epsg(4326)
            )  # 3182
            overlapping_crowns = sjoin(crowns, geo_central, how="inner")

            # skip firward if there are no crowns in a tile
            if crowns is not None:
                if overlapping_crowns.empty:
                    continue
                if len(overlapping_crowns) < threshold:
                    continue
            # here we are cropping the tiff to the bounding box of the tile we want
            coords = getFeatures(geo)
            # print("Coords:", coords)

            # define the tile as a mask of the whole tiff with just the bounding box
            # out_img, out_transform = mask(data, shapes=coords, crop=True)
            newbox = overlapping_crowns.total_bounds
            newbox = box(newbox[0], newbox[1], newbox[2], newbox[3])
            out_img, out_transform = mask(data, shapes=newbox, crop=True)
            # print('out transform:', out_transform)

            # This can be useful when reprojecting later as know the crs format to put it into
            # epsg_code = int(data.crs.data["init"][5:])
            # print(epsg_code)

            # copy the metadata then update it, the "nodata" and "dtype" where important as made larger
            # tifs have outputted tiles which were not just black
            out_meta = data.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_img.shape[1],
                    "width": out_img.shape[2],
                    "transform": out_transform,
                    "nodata": None,
                    "dtype": "uint8",
                }
            )
            # print('Out Meta:',out_meta)

            # Saving the tile as a new tiff, named by the origin of the tile. If tile appears blank in folder can show the image here and may
            # need to fix RGB data or the dtype
            # show(out_img)
            out_tif = out_dir + "tile_" + str(minx) + "_" + str(miny) + ".tif"
            with rasterio.open(out_tif, "w", **out_meta) as dest:
                dest.write(out_img)

            # read in the tile we have just saved
            clipped = rasterio.open(
                out_dir + "/tile_" + str(minx) + "_" + str(miny) + ".tif"
            )
            # read it as an array
            # show(clipped)
            arr = clipped.read()

            # each band of the tiled tiff is a colour!
            R = arr[0]
            G = arr[1]
            B = arr[2]

            # stack up the bands in an order appropriate for saving with cv2, then rescale to the correct 0-255 range for cv2

            rgb = np.dstack((B, G, R))  # BGR for cv2
            rgb_rescaled = rgb  # scale to image
            # print('rgb rescaled', rgb_rescaled)

            # save this as jpg or png...we are going for png...again, named with the origin of the specific tile
            # here as a naughty method
            cv2.imwrite(
                out_dir + "tile_" + str(minx) + "_" + str(miny) + ".png",
                rgb_rescaled,
            )

            # img = cv2.imread(
            #    "gdrive/MyDrive/JamesHirst/NY/LargeArea_images/naip_cayuga/naip_cayuga_tiled_by_me/tile_"
            #    + str(minx)
            #    + "_"
            #    + str(miny)
            #    + ".png"
            # )
            # print('png shape:', img.shape)
            if crowns is not None:
                # select the crowns that intersect the non-buffered central
                # section of the tile using the inner join
                # JB : a better solution would be to clip crowns to tile extent
                overlapping_crowns = sjoin(crowns, geo_central, how="inner")
                overlapping_crowns = overlapping_crowns.explode(index_parts=True)
                # print("Overlapping crowns:", overlapping_crowns)

                # translate to 0,0 to overlay on png
                # this now works as a universal approach.
                if minx == data.bounds[0] and miny == data.bounds[1]:
                    # print("We are in the bottom left!")
                    moved = overlapping_crowns.translate(-minx, -miny)
                elif miny == data.bounds[1]:
                    # print("We are on the bottom, but not bottom left")
                    moved = overlapping_crowns.translate(-minx + buffer, -miny)
                elif minx == data.bounds[0]:
                    # print("We are along the left hand side, but not bottom left!")
                    moved = overlapping_crowns.translate(-minx, -miny + buffer)
                else:
                    # print("We are in the middle!")
                    moved = overlapping_crowns.translate(-minx + buffer, -miny + buffer)
                # print("Moved coords:", moved)

                # scale to deal with the resolution
                scalingx = 1 / (data.transform[0])
                scalingy = -1 / (data.transform[4])
                moved_scaled = moved.scale(scalingx, scalingy, origin=(0, 0))
                # print(moved_scaled)

                impath = {
                    "imagePath": (
                        out_dir + "tile_" + str(minx) + "_" + str(miny) + ".png"
                    )
                }

                # save as a geojson, a format compatible with detectron2, again named by the origin of the tile.
                # If the box selected from the image is outside of the mapped region due to the image being on a slant
                # then the shp file will have no info on the crowns and hence will create an empty gpd Dataframe.
                # this causes an error so skip creating geojson. The training code will also ignore png so no problem.
                try:
                    filename = "./tile_" + str(minx) + "_" + str(miny) + ".geojson"
                    moved_scaled.to_file(
                        driver="GeoJSON",
                        filename=filename,
                    )
                    with open(filename, "r") as f:
                        shp = json.load(f)
                        shp.update(impath)
                    with open(filename, "w") as f:
                        json.dump(shp, f)
                except:
                    print("ValueError: Cannot write empty DataFrame to file.")
                    continue


if __name__ == "__main__":
    # Right let's test this first with Sepilok 10cm resolution, then I need to try it with 50cm resolution.
    img_path = "/content/drive/Shareddrives/detectreeRGB/benchmark/Ortho2015_benchmark/P4_Ortho_2015.tif"
    crown_path = "gdrive/MyDrive/JamesHirst/NY/Buffalo/Buffalo_raw_data/all_crowns.shp"
    out_dir = "./"
    # Read in the tiff file
    # data = img_data.open(img_path)
    # Read in crowns
    crowns = geopandas.read_file(crown_path)
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
    # resolution = 0.6 # in metres per pixel - @James Ball can you get this from the tiff?

    tile_data(data, buffer, tile_width, tile_height, out_dir, crowns)
