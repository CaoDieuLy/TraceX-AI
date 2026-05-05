import logging
from typing import Dict, List, Optional

import numpy as np
import torch

from ..core.types.bbox import BBox
from ..core.types.frame import Detection, Tracklet

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(
        self,
        tracking_method: str = "ocsort",
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ):
        self.tracking_method = tracking_method
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracked_objects: Dict[int, Tracklet] = {}
        self.next_id = 1
        self.frame_count = 0

        self._motion_history: Dict[int, List[np.ndarray]] = {}

    def update(self, detections_per_frame: List[List[Detection]]) -> List[Tracklet]:
        all_tracklets: List[Tracklet] = []

        for frame_detections in detections_per_frame:
            self.frame_count += 1
            frame_tracklets = self._update_frame(frame_detections)
            all_tracklets.extend(frame_tracklets)

        return all_tracklets

    def _update_frame(self, detections: List[Detection]) -> List[Tracklet]:
        if not detections:
            self._handle_unmatched_tracks()
            return []

        detection_boxes = [d.bbox for d in detections]
        matched, unmatched_dets, unmatched_tracks = self._match_detections_to_tracks(detection_boxes)

        frame_tracklets = []

        for det_idx, track_id in matched:
            detection = detections[det_idx]
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = Tracklet(
                    track_id=track_id,
                    detections=[],
                )

            self.tracked_objects[track_id].detections.append(detection)
            detection.track_id = track_id
            frame_tracklets.append(self.tracked_objects[track_id])

        for det_idx in unmatched_dets:
            new_id = self.next_id
            self.next_id += 1
            detection = detections[det_idx]
            detection.track_id = new_id

            new_tracklet = Tracklet(track_id=new_id, detections=[detection])
            self.tracked_objects[new_id] = new_tracklet
            self._motion_history[new_id] = []
            frame_tracklets.append(new_tracklet)

        return frame_tracklets

    def _match_detections_to_tracks(
        self,
        detection_boxes: List[BBox],
    ) -> tuple:
        if not self.tracked_objects:
            return [], list(range(len(detection_boxes))), []

        active_tracks = [
            (track_id, track)
            for track_id, track in self.tracked_objects.items()
            if len(track.detections) > 0
        ]

        if not active_tracks:
            return [], list(range(len(detection_boxes))), []

        cost_matrix = np.zeros((len(detection_boxes), len(active_tracks)))

        for d, det_bbox in enumerate(detection_boxes):
            for t, (track_id, track) in enumerate(active_tracks):
                last_bbox = track.detections[-1].bbox
                iou = det_bbox.iou_with(last_bbox)
                cost_matrix[d, t] = 1.0 - iou

        matches = []
        unmatched_detections = list(range(len(detection_boxes)))
        unmatched_tracks = list(range(len(active_tracks)))

        try:
            from scipy.optimize import linear_sum_assignment
            det_indices, track_indices = linear_sum_assignment(cost_matrix)

            for d, t in zip(det_indices, track_indices):
                if cost_matrix[d, t] < (1.0 - self.iou_threshold):
                    matches.append((d, active_tracks[t][0]))
                    if d in unmatched_detections:
                        unmatched_detections.remove(d)
                    if t in unmatched_tracks:
                        unmatched_tracks.remove(t)
        except ImportError:
            for d, det_bbox in enumerate(detection_boxes):
                for t, (track_id, track) in enumerate(active_tracks):
                    last_bbox = track.detections[-1].bbox
                    if det_bbox.iou_with(last_bbox) > self.iou_threshold:
                        matches.append((d, track_id))
                        if d in unmatched_detections:
                            unmatched_detections.remove(d)
                        break

        unmatched_track_ids = [active_tracks[t][0] for t in unmatched_tracks]

        return matches, unmatched_detections, unmatched_track_ids

    def _handle_unmatched_tracks(self) -> None:
        tracks_to_remove = []
        for track_id in self.tracked_objects:
            if track_id not in self._motion_history:
                self._motion_history[track_id] = []

            self._motion_history[track_id].append(self.frame_count)

            if len(self._motion_history[track_id]) > self.max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
            del self._motion_history[track_id]

    def get_all_tracklets(self) -> List[Tracklet]:
        return list(self.tracked_objects.values())

    def reset(self) -> None:
        self.tracked_objects.clear()
        self.next_id = 1
        self.frame_count = 0
        self._motion_history.clear()

    @property
    def info(self) -> dict:
        return {
            "tracking_method": self.tracking_method,
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "iou_threshold": self.iou_threshold,
            "active_tracks": len(self.tracked_objects),
        }
