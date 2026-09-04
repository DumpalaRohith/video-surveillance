"""
tracker.py
Wraps YOLOv8's built-in ByteTrack for multi-object tracking with
persistent IDs. Keeps ultralytics internals out of the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0


@dataclass
class Track:
    track_id: int
    bbox: tuple        # (x1, y1, x2, y2)
    confidence: float


class PersonTracker:
    # ByteTrack (bytetrack.yaml) is a *two-stage* associator: it matches
    # high-score boxes first, then tries to recover lost tracks by matching
    # remaining tracks against low-score boxes down to track_low_thresh
    # (0.1 by default) via IoU. If we only ever hand YOLO's detector boxes
    # above our own reporting threshold, that recovery stage never sees
    # anything and a person who's briefly occluded/blurred gets dropped and
    # re-spawned as a new ID instead of being re-associated. So we always
    # feed the detector down to this floor and apply `conf_threshold` as a
    # separate post-filter on what we report — not as the detector's conf arg.
    _ASSOC_CONF_FLOOR = 0.1

    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.25,
                 device: str = "cpu",
                 tracker_cfg: str = "bytetrack.yaml",
                 imgsz: int = 640):
        """
        conf_threshold: minimum confidence for a *tracked* box to be reported.
                        Applied after tracking, not as the detector's cutoff —
                        see _ASSOC_CONF_FLOOR above.
        tracker_cfg: 'bytetrack.yaml' (no re-id embedding, fast) or
                     'botsort.yaml' (adds appearance re-id, slower, better
                     at re-identifying people after occlusion/re-entry).
        imgsz: inference resolution. Raise this (e.g. 960-1280) for
               high-resolution, crowded scenes where distant people shrink to
               a handful of pixels at the default 640 and stop being detected.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self.tracker_cfg = tracker_cfg
        self.imgsz = imgsz

    def track(self, frame: np.ndarray) -> List[Track]:
        """Run detection + tracking on a single BGR frame. Call sequentially
        per-frame — the tracker keeps internal state across calls."""
        results = self.model.track(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=min(self.conf_threshold, self._ASSOC_CONF_FLOOR),
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker_cfg,
            persist=True,   # keep track state between calls
            verbose=False,
        )[0]

        tracks = []
        if results.boxes.id is None:
            return tracks  # no confirmed tracks this frame

        for box, tid in zip(results.boxes, results.boxes.id):
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue  # seen by the tracker for association, not reported
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tracks.append(Track(track_id=int(tid), bbox=(x1, y1, x2, y2), confidence=conf))
        return tracks

    def reset(self):
        """Clear tracker state — call this when starting a new video."""
        self.model.predictor = None