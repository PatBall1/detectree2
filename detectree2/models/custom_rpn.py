''' change iou metric to ioa in rpn nms process for better mask quality and combine these two in anchor matching for better box regression'''
''' a predictor using mask_iou for proprocessing'''

from typing import Tuple

import torch
import torchvision
from torch import Tensor
from torchvision.extension import _assert_has_ops
import logging
import math
from typing import List, Tuple, Union

from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances

from detectron2.structures import Boxes, Instances
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import nonzero_tuple
from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.structures.boxes import pairwise_intersection
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry
from detectron2.modeling.proposal_generator.rpn import RPN, build_rpn_head

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform, _dense_box_regression_loss
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling import build_model


@torch.jit.script_if_tracing
def move_device_like(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Tracing friendly way to cast tensor to another tensor's device. Device will be treated
    as constant during tracing, scripting the casting process as whole can workaround this issue.
    """
    return src.to(dst.device)
  
def _is_tracing():
    # (fixed in TORCH_VERSION >= 1.9)
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return torch.jit.is_tracing()        

def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute the IoA difference (intersection over boxes2 area - intersection over boxes2 area).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()  # [M]
    area1 = boxes1.area()  # [N]
    inter = pairwise_intersection(boxes1, boxes2)
    # handle empty boxes
    ioa_dif = torch.where(
        inter > 0, abs(inter / area2 - (inter.T / area1).T), torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa_dif

class Ioa_Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix and match_quality_matrix_ioa, 
    they characterize how well each (ground-truth, prediction)-pair match each other. For example,
    if the elements are boxes, this matrix may contain box intersection-over-union
    overlap values and intersection-over-area difference values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(
        self, thresholds: List[float], thresholds_ioa: List[float], labels: List[int], allow_low_quality_matches: bool = False
    ):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify predictions
                into levels.
            thresholds_ioa (list): a list of thresholds used to stratify predictions
                into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions with maximum match quality lower than high_threshold.
                See set_low_quality_matches_ for more details.
            For example,
                thresholds = [0.3, 0.5]
                thresholds_ioa = 0.6
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1 and
                thus will be ignored.
                All predictions with 0.5 <= iou and ioa_dif < 0.6 will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        assert thresholds[0] > 0
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        # Currently torchscript does not support all + generator
        assert all([low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:])])
        assert all([l in [-1, 0, 1] for l in labels])
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.thresholds_ioa = thresholds_ioa
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix, match_quality_matrix_ioa):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                two ioa difference between M ground-truth elements and N predicted
                elements. All elements must be >= 0 (due to the us of `torch.nonzero`
                for selecting indices in :meth:`set_low_quality_matches_`).
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is a matched
                ground-truth index in [0, M)
            match_labels (Tensor[int8]): a vector of length N, where pred_labels[i] indicates
                whether a prediction is a true or false positive or ignored
        """
        assert match_quality_matrix.dim() == 2
        if match_quality_matrix.numel() == 0:
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), 0, dtype=torch.int64
            )
            # When no gt boxes exist, we define IOU = 0 and therefore set labels
            # to `self.labels[0]`, which usually defaults to background class 0
            # To choose to ignore instead, can make labels=[-1,0,-1,1] + set appropriate thresholds
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),), self.labels[0], dtype=torch.int8
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        
        matched_vals_ioa = match_quality_matrix_ioa[matches, np.arange(0, match_quality_matrix.size(1))]

        match_labels = matches.new_full(matches.size(), 0, dtype=torch.int8)

        for (l, low, high) in zip(self.labels, self.thresholds[:-1], self.thresholds[1:]):
            low_high = (matched_vals >= low) & (matched_vals < high) & (matched_vals_ioa*l < self.thresholds_ioa)
            match_labels[low_high] = l

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.
        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        :paper:`Faster R-CNN`.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        _, pred_inds_with_highest_quality = nonzero_tuple(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # If an anchor was labeled positive only due to a low-quality match
        # with gt_A, but it has larger overlap with gt_B, it's matched index will still be gt_B.
        # This follows the implementation in Detectron, and is found to have no significant impact.
        match_labels[pred_inds_with_highest_quality] = 1


@PROPOSAL_GENERATOR_REGISTRY.register()
class custom_RPN(RPN):
  '''
  from detectron2-rpn.py 
  the main change is to use ioa instead of iou in nms and combine these two in anchor matching
  another change is that before nms will be processed for each class, but now images from all classes wil be processed together
  '''
  @configurable
  def __init__(self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Ioa_Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float,
        nms_thresh_union: float,
        min_box_size: float,
        anchor_boundary_thresh: float,
        loss_weight: Union[float, Dict[str, float]],
        box_reg_loss_type: str = "smooth_l1",
        smooth_l1_beta: float = 0.0,):
    super().__init__(in_features=in_features, head=head, anchor_generator=anchor_generator, 
                 anchor_matcher=anchor_matcher, box2box_transform=box2box_transform, batch_size_per_image=batch_size_per_image, 
                 positive_fraction=positive_fraction, pre_nms_topk=pre_nms_topk, post_nms_topk=post_nms_topk)
    self.nms_thresh_union = nms_thresh_union

  @classmethod
  def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    in_features = cfg.MODEL.RPN.IN_FEATURES
    ret = {
        "in_features": in_features,
        "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
        "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
        "nms_thresh_union": cfg.nms_thresh_union,
        "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
        "loss_weight": {
            "loss_rpn_cls": cfg.MODEL.RPN.LOSS_WEIGHT,
            "loss_rpn_loc": cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT * cfg.MODEL.RPN.LOSS_WEIGHT,
        },
        "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
        "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
        "box_reg_loss_type": cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE,
        "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
    }

    ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
    ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

    ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
    ret["anchor_matcher"] = Ioa_Matcher(
        cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOA_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
    )
    ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features]) 
    return ret

  @torch.jit.unused
  @torch.no_grad()
  def label_and_sample_anchors(
        self, anchors: List[Boxes], gt_instances: List[Instances]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.
        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            match_quality_matrix_ioa = retry_if_cuda_oom(pairwise_ioa)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix, match_quality_matrix_ioa)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes
    
  def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    nms_thresh_union: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
    training: bool,
):
    # here the box refinement has been done when proposals are inputted 
    num_images = len(image_sizes)
    device = (
        proposals[0].device
        if torch.jit.is_scripting()
        else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    )

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)


    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)

    # 3. For each image, run a per-level NMS, and choose topk results.
    results: List[Instances] = []
    for n, image_size in enumerate(image_sizes):
        print("----------------------------------------", image_size)
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
        boxes.clip(image_size)

        # filter empty boxes
        keep = boxes.nonempty(threshold=min_box_size)
        if _is_tracing() or keep.sum().item() != len(boxes):
            boxes, scores_per_img= boxes[keep], scores_per_img[keep]

        keep = custom_nms(boxes.tensor, scores_per_img, nms_thresh_union)

        keep = keep[:post_nms_topk]  # keep is already sorted

        res = Instances(image_size)
        res.proposal_boxes = boxes[keep]
        res.objectness_logits = scores_per_img[keep]
        results.append(res)
    return results
  
def custom_nms(P : torch.tensor ,scores: torch.tensor, thresh_iou_o : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes index, Shape: [ , 1]
    """
 
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
 
    # we extract the confidence scores as well
    scores = scores
 
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(idx)
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
         
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
 
        # find the intersection area
        inter = w*h
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 

        # find the interaction over S
        IoU_S = inter / areas[idx]

        # find the interaction over prediction
        IoU_P = inter / areas
 
        # keep the boxes with IoU less than thresh_iou
        mask = (IoU_S < thresh_iou_o)&(IoU_P < thresh_iou_o)
        order = order[mask]
     
    return keep
  
def custom_nms_mask(P : torch.tensor ,scores: torch.tensor, thresh_iou_o : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        masks: (tensor) The location preds for the image 
            along with the class predscores, Shape: [n,image_shape,image_shape].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes index, Shape: [ , 1]
    """
 
    # we turn masks into ndarray
    masks = P.reshape(len(P),-1,1)
 
    # we extract the confidence scores as well
    scores = scores
 
    # calculate area of every block in P
    areas = torch.sum(masks, axis = 1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(idx)
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
 
        # find the areas of BBoxes according the indices in order
        rem_areas = areas[order]

        # find the intersection area
        inter = torch.sum(masks[idx] * masks[order], axis=1)

        # find the interaction over S
        IoU_S = inter / areas[idx]

        # find the interaction over prediction
        IoU_P = inter / rem_areas
 
        # keep the masks with IoU less than thresh_iou
        mask = (IoU_S < thresh_iou_o)&(IoU_P < thresh_iou_o).reshape(-1,1)
        order = order.reshape(-1,1)[mask]
     
    return keep

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

            keep = custom_nms_mask(pred_masks ,scores, thresh_iou_o = self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)

            predictions['instances'].get_fields()['pred_boxes'] = boxes[keep]
            predictions['instances'].get_fields()['scores'] = scores[keep]
            predictions['instances'].get_fields()['pred_classes'] = pred_classes[keep]
            predictions['instances'].get_fields()['pred_masks'] = pred_masks[keep]

            return predictions
