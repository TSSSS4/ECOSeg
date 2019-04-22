import torch
import torch.nn as nn
from torch.nn import functional as F

from eco.config import config
from .pooler import Pooler
from .layer import make_conv3x3, Conv2d, ConvTranspose2d
from seg.utils.bounding_box import BoxList, boxlist_iou
from seg.utils.matcher import Matcher
from seg.utils.utils import cat
from seg.structures import resnet


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(nn.Module):
    def __init__(self, in_channels):
        super(ROIMaskHead, self).__init__()
        # FPN
        # self.feature_extractor = ROIMaskFeatureExtractor(in_channels)
        # self.predictor = ROIMaskPredictor(self.feature_extractor.out_channels)
        # self.post_processor = MaskPostProcessor()
        # self.loss_evaluator = make_roi_mask_loss_evaluator()
        # ResNet50
        self.feature_extractor = ResNet50Conv5ROIFeatureExtractor(in_channels)
        self.predictor = MaskRCNNC4Predictor(self.feature_extractor.out_channels)
        self.post_processor = MaskPostProcessor()
        self.loss_evaluator = make_roi_mask_loss_evaluator()

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            # proposals, positive_inds = keep_only_positive_boxes(proposals)

        x = self.feature_extractor(features, proposals)         # input: feature-maps and proposals, then FCN, output features
        mask_logits = self.predictor(x)                         # input FCN features [n,81,14,14], output masks[proposal_num,81,28,28]

        if not self.training:
            result = self.post_processor(mask_logits, proposals)    # keep one mask for each box according it's label
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


class ROIMaskFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(ROIMaskFeatureExtractor, self).__init__()

        resolution = config.roi_mask_pooler_resolution
        scales = config.roi_mask_pooler_scales
        sampling_ratio = config.roi_mask_pooler_sampling_scale
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        layers = config.roi_mask_conv_layers
        dilation = config.roi_mask_dialation

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.roi_mask_pooler_resolution
        scales = config.roi_mask_pooler_scales
        sampling_ratio = config.roi_mask_pooler_sampling_scale
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=resnet.BottleneckWithFixedBatchNorm,
            stages=(stage,),
            num_groups=1,
            width_per_group=64,
            stride_in_1x1=True,
            stride_init=None,
            res2_out_channels=256,
            dilation=1
        )

        self.pooler = pooler
        self.head = head
        self.out_channels = head.out_channels

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.head(x)
        return x


class ROIMaskPredictor(nn.Module):
    def __init__(self, in_channels):
        super(ROIMaskPredictor, self).__init__()
        num_classes = config.roi_box_num_class
        dim_reduced = config.roi_mask_conv_layers[-1]
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, in_channels):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = config.roi_box_num_class
        dim_reduced = config.roi_mask_conv_layers[-1]
        num_inputs = in_channels

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        x = F.relu(self.conv5_mask(x))
        return self.mask_fcn_logits(x)


class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes):
        """
        Arguments:
            x (Tensor): the mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_prob = x.sigmoid()     # (box_num=1, cls_num=2, 14, 14), cls(fg, bg)

        # select fg mask
        mask_prob = mask_prob[0, 1][:, None]

        boxes_per_image = 1
        mask_prob = mask_prob.split(boxes_per_image, dim=0)

        if self.masker:
            mask_prob = self.masker(mask_prob, boxes)

        results = []
        for prob, box in zip(mask_prob, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox.add_field("mask", prob)
            results.append(bbox)

        return results


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)


class ROIMaskLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            # matched_targets = self.match_targets_to_proposals(
            #     proposals_per_image, targets_per_image
            # )
            # # matched_idxs = matched_targets.get_field("matched_idxs")
            #
            # # mask scores are only computed on positive samples
            # positive_inds = torch.nonzero(torch.tensor([1])).squeeze(1)
            #
            # segmentation_masks = matched_targets.get_field("masks")
            # segmentation_masks = segmentation_masks[positive_inds]
            #
            # positive_proposals = proposals_per_image[positive_inds]
            #
            # masks_per_image = project_masks_on_boxes(
            #     segmentation_masks, positive_proposals, self.discretization_size
            # )

            segmentation_masks = targets_per_image.get_field("masks")
            positive_proposals = proposals_per_image
            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            masks.append(masks_per_image)

        return masks

    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        mask_targets = self.prepare_targets(proposals, targets)

        mask_targets = cat(mask_targets, dim=0)

        # positive_inds = torch.nonzero(labels > 0).squeeze(1)
        positive_inds = torch.tensor([0, 1])

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        # mask_logits(n, c, w, h)
        # why mask_logits[:,1]? all n proposal is positive, only 2 classes(bg, fg)
        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[:, 1], mask_targets
        )
        return mask_loss


def make_roi_mask_loss_evaluator():
    matcher = Matcher(
        config.roi_head_fg_iou_thresh,
        config.roi_head_bg_iou_thresh,
        allow_low_quality_matches=False,
    )

    loss_evaluator = ROIMaskLossComputation(
        matcher, config.roi_mask_pooler_resolution
    )

    return loss_evaluator


