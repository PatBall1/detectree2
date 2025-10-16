================================
Advanced Topics & "Pro Tips"
================================

This section contains a collection of advanced techniques and tips for expert users 
and special cases, such as training with multispectral data and handling class imbalance.

--------------------------------
Advanced Multispectral Training
--------------------------------

The process for training a multispectral model is similar to that for RGB data but there are some key steps that are
different. Data will be read from ``.tif`` files of 4 or more bands instead of the 3-band ``.png`` files.

First, ensure your data is tiled with ``mode="ms"`` and registered correctly.

The number of bands can be checked with rasterio:

.. code-block:: python

   import rasterio
   import glob

   folder_path = "/path/to/tilesMS/"
   img_paths = glob.glob(folder_path + "*.tif")
   img_path = img_paths[0]

   with rasterio.open(img_path) as dataset:
      num_bands = dataset.count
   print(f'The raster has {num_bands} bands.')


Due to the additional bands, the weights of the first convolutional layer (conv1) are modified to accommodate a
variable number of input channels. This is automatically done when ``imgmode`` is set to ``"ms"``. The ``setup_cfg`` function also automatically extends ``cfg.MODEL.PIXEL_MEAN`` and ``cfg.MODEL.PIXEL_STD`` to include the additional bands when ``num_bands`` is set to a value greater than 3.

.. code-block:: python

   from detectree2.models.train import MyTrainer, setup_cfg

   trains = ("ParacouMS_train",)
   tests = ("ParacouMS_val",)
   out_dir = "./ms_outputs"

   base_model = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

   # Set up the configuration for multispectral training
   cfg = setup_cfg(base_model, trains, tests, workers=2, eval_period=50,
                  max_iter=50000, out_dir=out_dir, imgmode="ms",
                  num_bands=num_bands)

With additional bands, you may need to reduce the number of images per batch if you encounter memory errors (e.g., ``CUDA out of memory``).

.. code-block:: python

   cfg.SOLVER.IMS_PER_BATCH = 1

   trainer = MyTrainer(cfg, patience=5) 
   trainer.resume_or_load(resume=False)
   trainer.train()

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pro Tip: Selective Band Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to experiment with using only a subset of your available spectral bands without creating new image files. This can be achieved by creating a custom data mapper that reads only specific bands from your ``.tif`` files.

First, define a list of the 1-based band indices you wish to use. Then, define a custom `FlexibleDatasetMapper` that incorporates this logic, and pass it to the training loader.

.. code-block:: python

   import detectree2.models.train as t
   import detectron2.data.transforms as T
   import rasterio
   import torch
   import numpy as np

   # Define which bands to read (1-based indices)
   only_read_bands = [4, 5, 6, 7]
   
   # You must update num_bands in the cfg to match the number of selected bands
   cfg.INPUT.NUM_IN_CHANNELS = len(only_read_bands)
   
   # Create a custom mapper class to read only specific bands
   class CustomBandMapper(t.FlexibleDatasetMapper):
       def __call__(self, dataset_dict):
           try:
               with rasterio.open(dataset_dict["file_name"]) as src:
                   # Read only the specified bands
                   img = src.read(indexes=only_read_bands)
               
               # Transpose to (H, W, C)
               img = np.transpose(img, (1, 2, 0)).astype("float32")

               aug_input = T.AugInput(img)
               transforms = self.augmentations(aug_input)
               img = aug_input.image
               dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

               if "annotations" in dataset_dict:
                   self._transform_annotations(dataset_dict, transforms, img.shape[:2])
               
               return dataset_dict
           except Exception as e:
               print(f"Error processing {dataset_dict.get('file_name', 'unknown')}: {e}")
               return None

   # Override the default train loader with one that uses the custom mapper
   t.FlexibleDatasetMapper = CustomBandMapper

   # Now, when you run trainer.train(), it will use only the bands specified.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pro Tip: Advanced Weight Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default method for adapting a 3-channel (RGB) pre-trained model to more input channels is to repeat the weights of the first three channels. The `detectree2` library provides a utility function to perform this weight adaptation.

For developers who need to adapt an existing model to a different number of input bands (e.g., for 4-band imagery), the `multiply_conv1_weights` function located in `detectree2.models.train` automatically copies weights of an existing model round-robin style. Without the call to this method, the model's weights would be initialized randomly across the whole input convolution layer.

---------------------------------
Advanced Multi-Class Techniques
---------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Handling Class Imbalance with Federated Loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-world datasets for multi-class problems are often imbalanced. To counteract this, you can enable Federated Loss, a technique that gives more weight to rare classes during training. You can enable this by setting a few parameters on the `cfg` object after it has been created.

.. code-block:: python

    # After creating the cfg object with setup_cfg(...)

    # Enable Federated Loss
    cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = True
    cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = True

    # This power parameter controls how much to weight the rare classes
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER = 0.7

    # You can also set the number of classes for the federated loss
    from detectree2.preprocessing.tiling import get_class_distribution
    class_distribution = get_class_distribution(tiles_paths[0], 5)
    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CLASSES = int(len(class_distribution) * 0.75)


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pro Tip: Transfer Learning with a Different Number of Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common scenario is fine-tuning a model that was pre-trained on a different number of classes than your new dataset. The `resume_or_load` method is designed to handle this automatically.

If a mismatch in the number of classes is detected between the checkpoint and the new model, **Detectron2** will adapt the architecture by randomly initializing the affected parts of the model. This provides a reasonable starting point for the new classes and allows training to proceed without crashing.

-------------------------
Advanced Fine-Tuning
-------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pro Tip: Advanced Fine-grained Layer Freezing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you load a pre-trained model, you may not want to retrain the entire network, especially if your own dataset is small. A powerful technique is to "freeze" parts of the network, making their weights non-trainable, and only fine-tune the higher-level layers. Here’s how you can apply this technique after creating the `trainer` object and before calling `trainer.train()`:

.. code-block:: python

   trainer = MyTrainer(cfg, patience=10)
   trainer.resume_or_load(resume=False)

   # --- Advanced: Freeze layers of the pre-trained backbone ---

   print("Applying custom layer freezing...")

   # Freeze the initial convolutional stem
   trainer.model.backbone.bottom_up.stem.freeze()

   # Freeze the blocks within the first residual stage (res2)
   for block in trainer.model.backbone.bottom_up.stages[0].children():
       block.freeze()

   print("Starting training with custom frozen layers.")
   trainer.train()

**Why is this useful?**

*   **Prevents Overfitting:** On small datasets, allowing the full network to train can cause it to "forget" the powerful general features it learned and instead memorize your small dataset. Freezing the early layers prevents this.
*   **Faster Training:** With fewer trainable parameters, each training iteration is faster.
*   **Experimentation:** It gives you a crucial tool for experimentation. If your new images are very different from the original training data, you might only freeze the `stem`. If they are very similar, you might freeze everything up to `res4`.
