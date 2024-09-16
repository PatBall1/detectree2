Tutorial (multiclass)
=====================

This tutorial goes through the steps of multiclass detection and 
delineation (e.g. species mapping, disease mapping). A guide to single 
class prediction is available
`here <https://patball1.github.io/detectree2/tutorial.html>`_ - this covers
more detail on the fundamentals of training and should be reviewed before this
tutorial. The multiclassprocess is slightly more intricate than single class
prediction as the classes need to be correctly encoded and caried throughout the pipeline.

The key steps are:

1. Preparing data
2. Training models
3. Evaluating model performance
4. Making landscape level predictions



Preparing data (RGB and multispectral)
--------------------------------------

Data can be prepared in a similar way to the single class case but the classes
and their order (mapping) need to be saved so that they can be accessed
consistently across training and prediction. The classes are saved in a json
file with the class names and their indices. The indices are used to encode 
the classes in the training.

.. code-block:: python

    import rasterio
    import geopandas as gpd

    # Load the data
    base_dir = "/content/drive/MyDrive/SHARED/detectree2"

    site_path = base_dir + "/data/Danum_lianas"

    # Set the path to the orthomosaic and the crown shapefile
    img_path = site_path + "/rgb/2017_50ha_Ortho_reproject.tif"
    crown_path = site_path + "/crowns/Danum_lianas_full2017.gpkg"

    # Here, we set the name of the output folder.
    # Set tiling parameters
    buffer = 30
    tile_width = 40
    tile_height = 40
    threshold = 0.6 
    appends = str(tile_width) + "_" + str(buffer) + "_" + str(threshold)

    out_dir = site_path + "/tilesClass_" + appends + "/"

    # Read in the tiff file
    data = rasterio.open(img_path)

    # Read in crowns (then filter by an attribute?)
    crowns = gpd.read_file(crown_path)
    crowns = crowns.to_crs(data.crs.data)
    print(crowns.head())

    class_column = 'status'

    # Record the classes and save the class mapping
    record_classes(
        crowns=crowns,  # Geopandas dataframe with crowns
        out_dir=out_dir,  # Output directory to save class mapping
        column=class_column,  # Column to be used for classes
        save_format='json'  # Choose between 'json' or 'pickle'
    )

    class_mapping_file = os.path.join(out_dir, "class_to_idx.json")

The class mapping has been saved in the output directory as a json file called
``class_to_idx.json``. This file can now be accessed to encode the classes in
training and prediction steps.

To tile the data, we call the ``tile_data`` function as we did in the single
class case except now we point to the column name of the classes.

.. code-block:: python

    # Tile the data
    tile_data(
        img_path=img_path,  # Path to the orthomosaic
        out_dir=out_dir,  # Output directory to save tiles
        buffer=buffer,  # Buffer around the crowns
        tile_width=tile_width,  # Width of the tiles
        tile_height=tile_height,  # Height of the tiles
        crowns=crowns,  # Geopandas dataframe with crowns
        threshold=threshold,  # Threshold for the buffer
        class_column=class_column,  # Column to be used for classes
    )
    
    # Split the data into training and validation sets 
    to_traintest_folders(
        tiles_folder=out_dir,  # Directory where tiles are saved
        out_folder=out_dir,  # Final directory for train/test data
        test_frac=0,  # Fraction of data to be used for testing
        folds=5,  # Number of folds (optional, can be set to 1 for no fold splitting)
        strict=False,  # Ensure no overlap between train/test tiles
        seed=42  # Set seed for reproducibility
    )


Training models
---------------

