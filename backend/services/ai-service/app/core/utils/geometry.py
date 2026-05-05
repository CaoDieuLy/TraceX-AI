import numpy as np
from typing import List, Tuple, Optional

from ..types.bbox import BBox


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    return BBox(
        x1=max(0, min(bbox.x1, width)),
        y1=max(0, min(bbox.y1, height)),
        x2=max(0, min(bbox.x2, width)),
        y2=max(0, min(bbox.y2, height)),
        confidence=bbox.confidence,
        class_id=bbox.class_id,
        class_name=bbox.class_name,
    )


def scale_bbox(bbox: BBox, scale_x: float, scale_y: float) -> BBox:
    return BBox(
        x1=bbox.x1 * scale_x,
        y1=bbox.y1 * scale_y,
        x2=bbox.x2 * scale_x,
        y2=bbox.y2 * scale_y,
        confidence=bbox.confidence,
        class_id=bbox.class_id,
        class_name=bbox.class_name,
    )


def compute_bbox_center(bbox: BBox) -> Tuple[float, float]:
    return (bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2


def compute_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_iou(bbox1: BBox, bbox2: BBox) -> float:
    xi1 = max(bbox1.x1, bbox2.x1)
    yi1 = max(bbox1.y1, bbox2.y1)
    xi2 = min(bbox1.x2, bbox2.x2)
    yi2 = min(bbox1.y2, bbox2.y2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = bbox1.area
    area2 = bbox2.area
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_center_distance(bbox1: BBox, bbox2: BBox) -> float:
    c1 = compute_bbox_center(bbox1)
    c2 = compute_bbox_center(bbox2)
    return compute_distance(c1, c2)


def nms(
    bboxes: List[BBox],
    iou_threshold: float = 0.45,
) -> List[int]:
    if not bboxes:
        return []

    indices = np.argsort([b.confidence for b in bboxes])[::-1]
    keep: List[int] = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        ious = [compute_iou(bboxes[current], bboxes[i]) for i in indices[1:]]
        indices = indices[1:][np.array(ious) < iou_threshold]

    return keep


def batch_extract_crops(
    image: np.ndarray,
    bboxes: List[BBox],
    target_size: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    crops = []
    for bbox in bboxes:
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

        if x2 > x1 and y2 > y1:
            crop = image[y1:y2, x1:x2]
            if target_size is not None:
                import cv2
                crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
            crops.append(crop)
        else:
            crops.append(np.zeros((64, 64, 3), dtype=np.uint8))

    return crops
