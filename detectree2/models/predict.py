"""Generate predictions."""
import json
import os
import random
from pathlib import Path

import cv2
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from detectree2.models.train import get_filenames, get_tree_dicts

from custom_nms import custom_nms_mask

# Code to convert RLE data from the output instances into Polygons,
# a small amout of info is lost but is fine.
# https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py <-- found here

class DefaultPredictor1:
    """
    intersection over area and nms is added here

    """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            boxes = predictions['instances'].get_fields()['pred_boxes']
            scores = predictions['instances'].get_fields()['scores']
            pred_classes = predictions['instances'].get_fields()['pred_classes']
            pred_masks = predictions['instances'].get_fields()['pred_masks']

            keep = custom_nms_mask(pred_masks ,scores, thresh_iou_o = 0.3)

            predictions['instances'].get_fields()['pred_boxes'] = boxes[keep]
            predictions['instances'].get_fields()['scores'] = scores[keep]
            predictions['instances'].get_fields()['pred_classes'] = pred_classes[keep]
            predictions['instances'].get_fields()['pred_masks'] = pred_masks[keep]

            return predictions

def predict_on_data(
    directory: str = "./",
    predictor=DefaultPredictor1,
    eval=False,
    save: bool = True,
    num_predictions=0,
):
    """Make predictions on tiled data.
    Predicts crowns for all png images present in a directory and outputs masks as jsons.
    """

    pred_dir = os.path.join(directory, "predictions")

    Path(pred_dir).mkdir(parents=True, exist_ok=True)

    if eval:
        dataset_dicts = get_tree_dicts(directory)
    else:
        dataset_dicts = get_filenames(directory)

    # Works out if all items in folder should be predicted on
    if num_predictions == 0:
        num_to_pred = len(dataset_dicts)
    else:
        num_to_pred = num_predictions

    for d in random.sample(dataset_dicts, num_to_pred):
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        # Creating the file name of the output file
        file_name_path = d["file_name"]
        # Strips off all slashes so just final file name left
        file_name = os.path.basename(os.path.normpath(file_name_path))
        file_name = file_name.replace("png", "json")
        output_file = os.path.join(pred_dir, f"Prediction_{file_name}")
        print(output_file)

        if save:
            # Converting the predictions to json files and saving them in the
            # specfied output file.
            evaluations = instances_to_coco_json(outputs["instances"].to("cpu"), d["file_name"])
            with open(output_file, "w") as dest:
                json.dump(evaluations, dest)


if __name__ == "__main__":
    print("something")
