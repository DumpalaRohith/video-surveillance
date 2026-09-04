"""
detector.py
Wraps YOLOv8 for person detection. Returns plain dataclasses so the rest
of the pipeline never depends on ultralytics internals.
"""

from dataclasses import dataclass
from typing import List
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class id for "person"


@dataclass
class Detection:
    bbox: tuple      # (x1, y1, x2, y2) in pixel coords
    confidence: float
    class_id: int = PERSON_CLASS_ID


class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.4,
                 device: str = "cpu"):
        """
        model_path: yolov8n (fastest) / yolov8s / yolov8m (more accurate, slower)
        device: 'cpu' or 'cuda' if you have a GPU
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame, return only 'person' boxes."""
        results = self.model.predict(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf))
        return detections