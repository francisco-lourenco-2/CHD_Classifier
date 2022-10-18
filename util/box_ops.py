# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, z0, x1, y1, z1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
         (x1 - x0), (y1 - y0), (z1 - z0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    volume1 = box_volume(boxes1)
    volume2 = box_volume(boxes2)

    lt = torch.max(boxes1[:, None, :3], boxes2[:, :3])  # [N,M,3]
    rb = torch.min(boxes1[:, None, 3:], boxes2[:, 3:])  # [N,M,3]

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    inter = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]  # [N,M]

    union = volume1[:, None] + volume2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    volume = wh[:, :, 0] * wh[:, :, 1]* wh[:, :, 2]

    return iou - (volume - union) / volume

def box_volume(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, z1, x2, y2, z2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1] * (boxes[:, 4] - boxes[:, 2]))

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W, D] where N is the number of masks, (H, W, D) are the spatial dimensions.

    Returns a [N, 6] tensors, with the boxes in xyzxyz format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    h, w, d = masks.shape[-3:]

    z = torch.arange(0, d, dtype=torch.float)
    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    z, y, x = torch.meshgrid(z, y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1)[0]
    z_min = z_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], 1)
