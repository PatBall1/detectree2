Tutorial
========

Tutorials for preparing data, training models, evaluating model performance and making landscape level predictions

Tiling
------

.. code-block:: php
   :caption: EXT:site_package/Configuration/TCA/Overrides/sys_template.php
   from detectree2.preprocessing.tiling import tile_data_train
   import rasterio
   import geopandas as gpd
   );

.. code-block:: php
   :caption: EXT:site_package/Configuration/TCA/Overrides/sys_template.php
   # Set up paths
   site_path = "/content/drive/Shareddrives/detectree2/data/Paracou"
   img_path = site_path + "/rgb/2016/Paracou_RGB_2016_10cm.tif"
   crown_path = site_path + "/crowns/220619_AllSpLabelled.gpkg"

   out_dir = site_path + '/tiles/'

   # Read in the tiff file
   data = rasterio.open(img_path)
   
   # Read in crowns (then filter by an attribute?)
   crowns = gpd.read_file(crown_path)
   crowns = crowns.to_crs(data.crs.data)
   
   # Set tiling parameters
   buffer = 30
   tile_width = 40
   tile_height = 40
   threshold = 0.6
   );