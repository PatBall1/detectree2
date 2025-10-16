===============================
In-Depth Guide: Prediction
===============================

This guide covers how to use a trained ``detectree2`` model to make predictions on large-scale orthomosaics.

----------------------------------
Generating Landscape Predictions
----------------------------------

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
can discard the partial crowns predicted at the edge of tiles.

.. code-block:: python
   
   # Path to site folder and orthomosaic
   site_path = "/path/to/data/BCI_50ha"
   img_path = site_path + "/rgb/2015.06.10_07cm_ORTHO.tif"
   tiles_path = site_path + "/tilespred/"

   # Location of trained model
   model_path = "/path/to/models/220629_ParacouSepilokDanum_JB.pth"

   # Specify tiling
   buffer = 30
   tile_width = 40
   tile_height = 40
   tile_data(img_path, tiles_path, buffer, tile_width, tile_height, dtype_bool = True)

.. warning::
   If tiles are outputting as blank images set ``dtype_bool = True`` in the ``tile_data`` function. This is a bug
   and we are working on fixing it. Avoid supplying crown polygons otherwise the function will run as if it is tiling
   for training.

To download a pre-trained model from the ``model_garden`` you can run ``wget`` on the package repo

.. code-block:: console
   
   !wget https://zenodo.org/records/15863800/files/250312_flexi.pth

Point to a trained model, set up the configuration state and make predictions on the tiles.

.. code-block:: python
   
   trained_model = "./230103_randresize_full.pth"
   cfg = setup_cfg(update_model=trained_model)
   predict_on_data(tiles_path, predictor=DefaultPredictor(cfg))

Once the predictions have been made on the tiles, it is necessary to project them back into geographic space.

.. code-block:: python
   
   project_to_geojson(tiles_path, tiles_path + "predictions/", tiles_path + "predictions_geo/")

To create a useful output it is necessary to stitch the crowns together while handling overlaps in the buffer.
Invalid geometries may arise when converting from a mask to a polygon - it is usually best to simply remove these.
Cleaning the crowns will remove instances where there is large overlaps between predicted crowns (removing the
predictions with lower confidence).

.. code-block:: python
   
   crowns = stitch_crowns(tiles_path + "predictions_geo/", 1)
   clean = clean_crowns(crowns, 0.6, confidence=0) # set a confidence>0 to filter out less confident crowns

By default the ``clean_crowns`` function will remove crowns with a confidence of less than 20%. The above 'clean' crowns
includes crowns of all confidence scores (0%-100%) as ``confidence=0``. It is likely that crowns with very low
confidence will be poor quality so it is usually preferable to filter these out. A suitable threshold can be determined
by eye in QGIS or implemented as single line in Python. ``Confidence_score`` is a column in the ``crowns`` GeoDataFrame
and is considered a tunable parameter.

.. code-block:: python
   
   clean = clean[clean["Confidence_score"] > 0.5] # step included for illustration - can be done in clean_crowns func

The outputted crown polygons will have many vertices because they are generated from a mask which is pixelwise. If you
will need to edit the crowns in QGIS it is best to simplify them to a reasonable number of vertices. This can be done
with ``simplify`` method. The ``tolerance`` will determine the coarseness of the simplification it has the same units as
the coordinate reference system of the GeoSeries (meters when working with UTM).

.. code-block:: python
   
   clean = clean.set_geometry(clean.simplify(0.3))

Once we're happy with the crown map, save the crowns to file.

.. code-block:: python
   
   clean.to_file(site_path + "/crowns_out.gpkg")

-----------------------------------
Landscape Predictions (Multi-Class)
-----------------------------------

COMING SOON
