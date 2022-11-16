Tutorial
========

A tutorial for:

1. preparing data
2. training models
3. evaluating model performance
4. making landscape level predictions

Before getting started ensure detectree2 is installed either through

.. code-block:: console

   (.venv) $pip install git+https://github.com/PatBall1/detectree2.git

or

.. code-block:: console

   (.venv) $conda install detectree2 -c ma595

To train a model you will need an orthomosaic (as ``<orthmosaic.tif``) and

corresponding tree crown polgons that are readable by Geopandas
(e.g. ``<crowns_polygon>.gpkg``, ``<crowns_polygon>.shp``). For the best
results, manual crowns should be supplied as dense clusters rather than
sparsely scattered across in the landscape


If you would just like to make predictions on an orthomosaic with a pre-trained
model from the ``model_garden``, skip to part 4.


Preparing data
--------------

An example of the recommended file structure when training a new model is as follows:

.. code-block:: bash

   ├── Danum                                       (site directory)
   │   ├── rgb
   │   │   └── Dan_2014_RGB_project_to_CHM.tif     (RGB orthomosaic in local UTM CRS)
   │   └── crowns
   │       └── Danum.gpkg                          (Crown polygons readable by geopandas e.g. Geopackage, shapefile)
   │ 
   └── Paracou                                     (site directory)
       ├── rgb                                     
       │   ├── Paracou_RGB_2016_10cm.tif           (RGB orthomosaic in local UTM CRS)
       │   └── Paracou_RGB_2019.tif                (RGB orthomosaic in local UTM CRS)
       └── crowns
           └── UpdatedCrowns8.gpkg                 (Crown polygons readable by geopandas e.g. Geopackage, shapefile)

Here we have two site available to train on. Multiple site directories can be 
included in the training and testing phase but only a single site directory is 
required.
If available, several RGB orthomosaics can be included in a single site
directory (see e.g ``Paracou -> RGB``).

We will call functions to from ``detectree2``'s' tiling and training modules.

.. code-block:: python
   
   from detectree2.preprocessing.tiling import tile_data_train, to_traintest_folders
   from detectree2.models.train import register_train_data, MyTrainer, setup_cfg
   import rasterio
   import geopandas as gpd

Set up the paths for the orthomosaic and corresponding manual crown data.

.. code-block:: python
   
   # Set up input paths
   site_path = "/content/drive/Shareddrives/detectree2/data/Paracou"
   img_path = site_path + "/rgb/2016/Paracou_RGB_2016_10cm.tif"
   crown_path = site_path + "/crowns/220619_AllSpLabelled.gpkg"

   out_dir = site_path + '/tiles/'

   # Read in the tiff file
   data = rasterio.open(img_path)
   
   # Read in crowns (then filter by an attribute?)
   crowns = gpd.read_file(crown_path)
   crowns = crowns.to_crs(data.crs.data)

Set up the tiling parameters.

The tile size will depend on:
- The resolution of your imagery

.. code-block:: python

   # Set tiling parameters
   buffer = 30
   tile_width = 40
   tile_height = 40
   threshold = 0.6


Next we tile the data

.. code-block:: python
   
   tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns, threshold)


Send geojsons to train folder (with sub-folders for k-fold cross validation)
 and test folder. 
The approximate proportion of data to reserve for testing.
Automatically removes training tiles that overlap with test tiles, ensuring
spatial separation 

.. code-block:: python
   
   to_traintest_folders(data_folder, out_dir, test_frac=0.15, folds=5)

Are data is now tiled and partitioned for training and model evaluation

.. code-block:: bash
   
   └── Danum                                       (site directory)
       ├── rgb
       │   └── Dan_2014_RGB_project_to_CHM.tif     (RGB orthomosaic in local UTM CRS)
       ├── crowns
       │   └── Danum.gpkg
       └── tiles                                   (tile directory)
           ├── train
           │   ├── fold_1                          (train fold folder)
           │   ├── fold_2                          (train fold folder)
           │   └── ...
           └── test                                (test data folder)
 

Training a model
----------------

Register the training data. It is possible to set a validation fold for model
evaluation

.. code-block:: python
   
   train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles/train/"
   register_train_data(train_location, "Paracou", val_fold)


Supply the ``base_model`` from Detectron2's  ``model_zoo``

.. code-block:: python
   
   # Set the base (pre-trained) model from the detectron2 model_zoo
   base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
      
   # trained_model = "/content/drive/Shareddrives/detectree2/models/220629_ParacouSepilokDanum_JB.pth"
   trains = ("Paracou_train", "Paracou2019_train", "ParacouUAV_train", "Danum_train", "SepilokEast_train", "SepilokWest_train")
   tests = ("Paracou_val", "Paracou2019_val", "ParacouUAV_val", "Danum_val", "SepilokEast_val", "SepilokWest_val")
   
   #trains = ("Paracou_train", "Paracou2019_train")
   #tests = ("Paracou_val", "Paracou2019_val")
   out_dir = "/content/drive/Shareddrives/detectree2/220809_train_outputs"
   
   cfg = setup_cfg(base_model, trains, tests, workers = 4, eval_period=100, max_iter=3000, out_dir=out_dir) # update_model arg can be used to load in trained  model

Evaluating model performance
----------------------------

Coming soon! See Colab notebook for example routine (detectree2/notebooks/colab/evaluationJB.ipynb).

Generating landscape predictions
--------------------------------

Call necessary functions.

.. code-block:: python
   
   from detectree2.preprocessing.tiling import tile_data
   from detectree2.models.train import MyTrainer, setup_cfg


Start by tiling up the entire orthomosaic so that a crown map can be made for the entire landscape. Tiles should be 
approximately the same size as those trained on (typically ~ 100 m).

.. code-block:: python
   
   # Path to site folder and orthomosaic
   site_path = "/content/drive/Shareddrives/detectree2/data/BCI_50ha"
   img_path = site_path + "/rgb/2015.06.10_07cm_ORTHO.tif"
   tiles_path = site_path + "/tilespred/"
   # Location of pre-trained model
   model_path = "/content/drive/Shareddrives/detectree2/models/220629_ParacouSepilokDanum_JB.pth"

   # Specify tiling
   buffer = 30
   tile_width = 40
   tile_height = 40
   tile_data(data, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)


Point to a trained model, set up the configuration state and make predictions on the tile.

.. code-block:: python
   
   trained_model = "/content/drive/Shareddrives/detectree2/models/220723_withParacouUAV.pth"
   cfg = setup_cfg(update_model=trained_model)
   predict_on_data(tiles_path, DefaultPredictor(cfg))

Once the predictions have been made on the tiles, it is necessary to project them back into geographic space

.. code-block:: python
   
   project_to_geojson(data, tiles_path + "predictions_geo/", tiles_path + "predictions/")

To create a useful outputs it is necessary to stitch the crowns together while handling overlaps in the buffer.
Invalid geometries may arise when converting from a mask to a polygon - it is usually best to simply remove these.
Cleaning the crowns will remove instances where there is large overlaps between predicted crowns (removing the
predictions with lower confidence).

.. code-block:: python
   
   project_to_geojson(data, tiles_path + "predictions_geo/", tiles_path + "predictions/")
   crowns = crowns[crowns.is_valid]
   crowns = clean_crowns(crowns, 0.6)

Once we're happy with the crown map, save the crowns to file.

.. code-block:: python
   
   crowns.to_file("/content/drive/Shareddrives/detectree2/data/" + name + "/crowns_out.gpkg")

