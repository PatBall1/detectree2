Tutorial
========

A tutorial for:

1. Preparing data
2. Training models
3. Evaluating model performance
4. Making landscape level predictions

Before getting started ensure ``detectree2`` is installed through

.. code-block:: console

   (.venv) $pip install git+https://github.com/PatBall1/detectree2.git


To train a model you will need an orthomosaic (as ``<orthmosaic>.tif``) and 
corresponding tree crown polgons that are readable by Geopandas
(e.g. ``<crowns_polygon>.gpkg``, ``<crowns_polygon>.shp``). For the best
results, manual crowns should be supplied as dense clusters rather than
sparsely scattered across in the landscape


If you would just like to make predictions on an orthomosaic with a pre-trained
model from the ``model_garden``, skip to part 4 (Generating landscape predictions).


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

Here we have two sites available to train on (Danum and Paracou). Several site directories can be 
included in the training and testing phase (but only a single site directory is required).
If available, several RGB orthomosaics can be included in a single site directory (see e.g ``Paracou -> RGB``).

We call functions to from ``detectree2``'s tiling and training modules.

.. code-block:: python
   
   from detectree2.preprocessing.tiling import tile_data_train, to_traintest_folders
   from detectree2.models.train import register_train_data, MyTrainer, setup_cfg
   import rasterio
   import geopandas as gpd

Set up the paths to the orthomosaic and corresponding manual crown data.

.. code-block:: python
   
   # Set up input paths
   site_path = "/content/drive/Shareddrives/detectree2/data/Paracou"
   img_path = site_path + "/rgb/2016/Paracou_RGB_2016_10cm.tif"
   crown_path = site_path + "/crowns/220619_AllSpLabelled.gpkg"

   # Read in the tiff file
   data = rasterio.open(img_path)
   
   # Read in crowns (then filter by an attribute if required)
   crowns = gpd.read_file(crown_path)
   crowns = crowns.to_crs(data.crs.data) # making sure CRS match

Set up the tiling parameters.

The tile size will depend on:

* The resolution of your imagery.
* Available computational resources.
* The detail required on the crown outline.
* If using a pre-trained model, the tile size used in training should roughly match the tile size of predictions.

.. code-block:: python

   # Set tiling parameters
   buffer = 30
   tile_width = 40
   tile_height = 40
   threshold = 0.6
   appends = str(tile_width) + "_" + str(buffer) + "_" + str(threshold) # this helps keep file structure organised
   out_dir = site_path + "/tiles_" + appends + "/"

The total tile size here is 100 m x 100 m (a 40 m x 40 m core area with a surrounding 30 m buffer that overlaps with
surrounding tiles). Including a buffer is recommended as it allows for tiles that include more training crowns.

Next we tile the data. The ``tile_data_train`` function will only retain tiles that contain more than the given
``threshold`` coverage of training data (here 60%). This helps to reduce the chance that the network is trained with
tiles that contain a large number of unlabelled crowns (which would reduce its sensitivity).

.. code-block:: python
   
   tile_data_train(data, out_dir, buffer, tile_width, tile_height, crowns, threshold)

.. warning::
   If tiles are outputing as blank images set ``dtype_bool = True`` in the ``tile_data_train`` function. This is a bug
   and we are working on fixing it.

.. note::
   You will want to relax the ``threshold`` value if your trees are sparsely distributed across your landscape.
   Remember, ``detectree2`` was initially designed for dense, closed canopy forests so some of the default assumptions 
   will reflect that.

Send geojsons to train folder (with sub-folders for k-fold cross validation) and test folder.

.. code-block:: python
   
   data_folder = out_dir # data_folder is the folder where the .png, .tif, .geojson tiles have been stored
   to_traintest_folders(data_folder, out_dir, test_frac=0.15, folds=5)

.. note::
   The ``to_traintest_folders`` function automatically removes training/validation geojsons that overlap with test
   tiles, ensuring strict spatial separation of the test data. However, this can remove a significant proportion of the
   data available to train on so if validation accuracy is a sufficient test of model performance ``test_frac`` can be
   set to ``0``. Alternatively, just set a ``test_frac`` value that is smaller than you might otherwise have put.


The data has now been tiled and partitioned for model training, tuning and evaluation.

.. code-block::
   
   └── Danum                                       (site directory)
       ├── rgb
       │   └── Dan_2014_RGB_project_to_CHM.tif     (RGB orthomosaic in local UTM CRS)
       ├── crowns
       │   └── Danum.gpkg
       └── tiles                                   (tile directory)
           ├── train
           │   ├── fold_1                          (train/val fold folder)
           │   ├── fold_2                          (train/val fold folder)
           │   └── ...
           └── test                                (test data folder)
 

Training a model
----------------

Before training can commence, it is necessary to register the training data. It is possible to set a validation fold for
model evaluation (which can be helpful for tuning models). The validation fold can be changed over different training 
steps to expose the model to the full range of available training data. Register as many different folders as necessary

.. code-block:: python
   
   train_location = "/content/drive/Shareddrives/detectree2/data/Danum/tiles_" + appends + "/train/"
   register_train_data(train_location, 'Danum', val_fold=5)

   train_location = "/content/drive/Shareddrives/detectree2/data/Paracou/tiles_" + appends + "/train/"
   register_train_data(train_location, "Paracou", val_fold=5) 

The data will be registered as ``<name>_train`` and ``<name>_val`` (or ``Paracou_train`` and ``Paracou_val`` in the
above example). It will be necessary to supply these registation names below...

We must supply a ``base_model`` from Detectron2's  ``model_zoo``. This loads a backbone that has been pre-trained which
saves us the pain of training a model from scratch. We are effectively transferring this model and (re)training it on
our problem for the sake of time and efficiency. The `trains` and `tests` variables containing the registered datasets 
should be tuples containing strings. If just a single site is being used a comma should still be supplied (e.g. 
``trains = ("Paracou_train",)``) otherwise the data loader will malfunction.

.. code-block:: python
   
   # Set the base (pre-trained) model from the detectron2 model_zoo
   base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
      
   trains = ("Paracou_train", "Danum_train", "SepilokEast_train", "SepilokWest_train") # Registered train data
   tests = ("Paracou_val", "Danum_val", "SepilokEast_val", "SepilokWest_val") # Registered validation data
   
   out_dir = "/content/drive/Shareddrives/detectree2/220809_train_outputs"
   
   cfg = setup_cfg(base_model, trains, tests, workers = 4, eval_period=100, max_iter=3000, out_dir=out_dir) # update_model arg can be used to load in trained  model


Alternatively, it is possible to train from one of ``detectree2``'s pre-trained models. This is normally recommended and
especially useful if you only have limited training data available. To retrieve the model from the repo's
``model_garden`` run e.g.:

.. code-block:: python

   !wget https://github.com/PatBall1/detectree2/raw/master/model_garden/230103_randresize_full.pth

Then set up the configurations as before but with the trained model also supplied:

.. code-block:: python

   # Set the base (pre-trained) model from the detectron2 model_zoo
   base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

   # Set the updated model weights from the detectree2 pre-trained model
   trained_model = "./230103_randresize_full.pth"
      
   trains = ("Paracou_train", "Danum_train", "SepilokEast_train", "SepilokWest_train") # Registered train data
   tests = ("Paracou_val", "Danum_val", "SepilokEast_val", "SepilokWest_val") # Registered validation data
   
   out_dir = "/content/drive/Shareddrives/detectree2/220809_train_outputs"
   
   cfg = setup_cfg(base_model, trains, tests, trained_model, workers = 4, eval_period=100, max_iter=3000, out_dir=out_dir) # update_model arg used to load in trained model

.. note::

   You may want to experiment with how you set up the ``cfg``. The variables can make a big difference to how quickly 
   model training will converge given the particularities of the data supplied and computational resources available.

Once we are all set up, we can get commence model training. Training will continue until a specified number of
iterations (``max_iter``) or until model performance is no longer improving ("early stopping" via ``patience``).
Training outputs, including model weights and training metrics, will be stored in ``out_dir``.

.. code-block::

   trainer = MyTrainer(cfg, patience = 5) 
   trainer.resume_or_load(resume=False)
   trainer.train()

.. note::

   Early stopping is implemented and will be triggered by a sustained failure to improve on the performance of
   predictions on the validation fold. This is measured as the AP50 score of the validation predictions.


Evaluating model performance
----------------------------

Coming soon! See Colab notebook for example routine (``detectree2/notebooks/colab/evaluationJB.ipynb``).

Generating landscape predictions
--------------------------------

Here we call the necessary functions.

.. code-block:: python
   
   from detectree2.preprocessing.tiling import tile_data
   from detectree2.models.outputs import project_to_geojson, stitch_crowns, clean_crowns
   from detectree2.models.predict import predict_on_data
   from detectree2.models.train import setup_cfg
   from detectron2.engine import DefaultPredictor
   import rasterio


Start by tiling up the entire orthomosaic so that a crown map can be made for the entire landscape. Tiles should be 
approximately the same size as those trained on (typically ~ 100 m). A buffer (here 30 m) should be included so that we 
can discard partial the crowns predicted at the edge of tiles.

.. code-block:: python
   
   # Path to site folder and orthomosaic
   site_path = "/content/drive/Shareddrives/detectree2/data/BCI_50ha"
   img_path = site_path + "/rgb/2015.06.10_07cm_ORTHO.tif"
   tiles_path = site_path + "/tilespred/"
   # Read in the geotiff
   data = rasterio.open(img_path)
   # Location of trained model
   model_path = "/content/drive/Shareddrives/detectree2/models/220629_ParacouSepilokDanum_JB.pth"

   # Specify tiling
   buffer = 30
   tile_width = 40
   tile_height = 40
   tile_data(data, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)

.. warning::
   If tiles are outputing as blank images set ``dtype_bool = True`` in the ``tile_data`` function. This is a bug
   and we are working on fixing it.

To download a pre-trained model from the ``model_garden`` you can run ``wget`` on the package repo

.. code-block:: python
   
   !wget https://github.com/PatBall1/detectree2/raw/master/model_garden/230103_randresize_full.pth


Point to a trained model, set up the configuration state and make predictions on the tiles.

.. code-block:: python
   
   trained_model = "./230103_randresize_full.pth"
   cfg = setup_cfg(update_model=trained_model)
   predict_on_data(tiles_path, DefaultPredictor(cfg))

Once the predictions have been made on the tiles, it is necessary to project them back into geographic space.

.. code-block:: python
   
   project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")

To create a useful outputs it is necessary to stitch the crowns together while handling overlaps in the buffer.
Invalid geometries may arise when converting from a mask to a polygon - it is usually best to simply remove these.
Cleaning the crowns will remove instances where there is large overlaps between predicted crowns (removing the
predictions with lower confidence).

.. code-block:: python
   
   crowns = stitch_crowns(tiles_path + "predictions_geo/", 1, confidence=0)
   crowns = crowns[crowns.is_valid]
   crowns = clean_crowns(crowns, 0.6)

By default the ``clean_crowns`` function will remove crowns with a condidence of less than 20%. The above 'clean' crowns
includes crowns of all confidence scores (0%-100%) as ``condidence=0``. It is likely that crowns with very low
confidence will be poor quality so it is usually preferable to filter these out. A suitable threshold can be determined
by eye in QGIS or implemented as single line in Python. ``Confidence_score`` is a column in the ``crowns`` GeoDataFrame
and is considered a tunable parameter.

.. code-block:: python
   
   crowns = crowns[crowns["Confidence_score"] > 0.5]

The outputted crown polygons will have many vertices because they are generated from a mask which is pixelwise. If you
will need to edit the crowns in QGIS it is best to simplify them to a reasonable number of vertices. This can be done
with ``simplify`` method. The ``tolerance`` will determine the coarseness of the simplification it has the same units as
the coordinate reference system of the GeoSeries (meters when working with UTM).

.. code-block:: python
   
   clean = clean.set_geometry(crowns.simplify(0.3))

Once we're happy with the crown map, save the crowns to file.

.. code-block:: python
   
   crowns.to_file(site_path + "/crowns_out.gpkg")

