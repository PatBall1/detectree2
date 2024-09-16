import cv2
import detectron2.data.transforms as T
import numpy as np
import rasterio
import torch
from detectron2.structures import BitMasks, BoxMode, Instances
from torch.utils.data import Dataset


class CustomTIFFDataset(Dataset):
    def __init__(self, annotations, transforms=None):
        """
        Args:
            annotations (list): List of dictionaries containing image file paths and annotations.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load the TIFF image
        img_info = self.annotations[idx]
        with rasterio.open(img_info['file_name']) as src:
            # Read all bands (assuming they are all needed)
            image = src.read()
            # Normalize or rescale if necessary
            image = image.astype(np.float32) / 255.0  # Example normalization
            # If the number of bands is not 3, reduce to 3 or handle accordingly
            #if image.shape[0] > 3:
            #    image = image[:3, :, :]  # Taking the first 3 bands (e.g., RGB)
            # Convert to HWC format expected by Detectron2
            #image = np.transpose(image, (1, 2, 0))

        # Prepare annotations (this part needs to be adapted to your specific annotations)
        target = {
            "image_id": idx,
            "annotations": img_info["annotations"],
            "width": img_info["width"],
            "height": img_info["height"],
        }

        if self.transforms is not None:
            augmentations = T.AugmentationList(self.transforms)
            image, target = augmentations(image, target)

        # Convert to Detectron2-compatible format
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        instances = self.get_detectron_instances(target)

        return image, instances

    def get_detectron_instances(self, target):
        """
        Converts annotations into Detectron2's format.
        This example assumes annotations are in COCO format, and you'll need to adapt it for your needs.
        """
        boxes = [obj["bbox"] for obj in target["annotations"]]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        
        # Create BitMasks from the binary mask data (assuming the mask is a binary numpy array)
        masks = [obj["segmentation"] for obj in target["annotations"]]  # Replace with actual mask loading
        masks = BitMasks(torch.stack([torch.from_numpy(mask) for mask in masks]))

        instances = Instances(
            image_size=(target["height"], target["width"]),
            gt_boxes=boxes,
            gt_classes=torch.tensor([obj["category_id"] for obj in target["annotations"]], dtype=torch.int64),
            gt_masks=masks
        )
        return instances
