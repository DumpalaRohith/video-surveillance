"""
pipeline.py
Orchestrates: video read -> detect+track -> zone events -> annotate -> write.
"""

import cv2
import csv
import json
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import asdict

from src.tracker import PersonTracker
from src.events import EventDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# BGR colors
COLOR_NORMAL = (0, 255, 0)       # green — not in any zone
COLOR_INTRUSION = (0, 0, 255)    # red — currently inside a restricted zone
COLOR_LOITER_WARN = (0, 165, 255)   # orange — in loitering zone, under threshold
COLOR_LOITER_ALERT = (0, 0, 255)    # red — loitering threshold exceeded
COLOR_FLASH = (255, 0, 255)      # magenta — the exact moment an event fires


class SurveillancePipeline:
    def __init__(self, zones_path: str, model_path: str = "yolov8n.pt",
                 conf_threshold: float = 0.25, device: str = "cpu",
                 tracker_cfg: str = "bytetrack.yaml", imgsz: int = 640):
        self.tracker = PersonTracker(model_path=model_path,
                                      conf_threshold=conf_threshold,
                                      device=device,
                                      tracker_cfg=tracker_cfg,
                                      imgsz=imgsz)
        self.zones_path = zones_path
        self.event_detector = None

    def run(self, video_path: str, output_dir: str):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.tracker.reset()
        self.event_detector = EventDetector(self.zones_path, fps=fps)

        out_video_path = output_dir / f"{video_name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        # Raw per-frame bounding boxes + confidence for every track, written
        # incrementally (not buffered in memory) so this scales to long videos.
        tracks_path = output_dir / f"{video_name}_tracks.csv"
        tracks_file = open(tracks_path, "w", newline="")
        tracks_writer = csv.writer(tracks_file)
        tracks_writer.writerow(["frame", "timestamp", "track_id",
                                 "x1", "y1", "x2", "y2", "confidence"])

        all_events = []
        frame_number = 0
        start_time = time.time()

        logger.info(f"Processing {video_path} ({total_frames} frames @ {fps:.1f} fps)")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tracks = self.tracker.track(frame)
            timestamp = frame_number / fps
            for t in tracks:
                x1, y1, x2, y2 = t.bbox
                tracks_writer.writerow([frame_number, f"{timestamp:.3f}", t.track_id,
                                         f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                                         f"{t.confidence:.4f}"])

            events = self.event_detector.update(frame_number, tracks)
            all_events.extend(events)

            for e in events:
                logger.info(f"[EVENT] frame={e.frame_number} id={e.track_id} "
                            f"{e.event_type} in '{e.zone_name}' (conf={e.confidence:.2f})")

            status = self.event_detector.get_status(frame_number, tracks)
            annotated = self._annotate(frame, tracks, status)
            writer.write(annotated)

            frame_number += 1
            if frame_number % 50 == 0:
                logger.info(f"  frame {frame_number}/{total_frames}")

        cap.release()
        writer.release()
        tracks_file.close()

        elapsed = time.time() - start_time
        processing_fps = frame_number / elapsed if elapsed > 0 else 0
        logger.info(f"Done: {frame_number} frames in {elapsed:.1f}s "
                     f"({processing_fps:.1f} fps processing speed)")

        events_path = output_dir / f"{video_name}_events.json"
        with open(events_path, "w") as f:
            json.dump([asdict(e) for e in all_events], f, indent=2)
        logger.info(f"Wrote {len(all_events)} events to {events_path}")
        logger.info(f"Wrote per-frame tracks to {tracks_path}")
        logger.info(f"Wrote annotated video to {out_video_path}")

        return {
            "annotated_video": str(out_video_path),
            "events_log": str(events_path),
            "tracks_log": str(tracks_path),
            "total_events": len(all_events),
            "processing_fps": round(processing_fps, 2),
        }

    def _annotate(self, frame, tracks, status):
        annotated = frame.copy()

        # zone outlines
        for zone in self.event_detector.zones:
            pts = [(int(x), int(y)) for x, y in zone.polygon.exterior.coords]
            color = COLOR_INTRUSION if zone.zone_type == "intrusion" else COLOR_LOITER_WARN
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32)],
                          isClosed=True, color=color, thickness=2)
            cv2.putText(annotated, zone.name, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        # tracks — color/label driven by live zone status, not just this-frame events
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            zone_statuses = status.get(t.track_id, [])

            color = COLOR_NORMAL
            label = f"ID {t.track_id} ({t.confidence:.2f})"
            thickness = 2

            if zone_statuses:
                zs = zone_statuses[0]  # a person is rarely in >1 zone at once

                if zs.just_triggered:
                    color = COLOR_FLASH
                    thickness = 4
                    label = f"ID {t.track_id} ALERT: {zs.zone_type.upper()}"
                elif zs.zone_type == "intrusion":
                    color = COLOR_INTRUSION
                    label = f"ID {t.track_id} IN RESTRICTED ZONE"
                elif zs.zone_type == "loitering":
                    if zs.threshold_seconds and zs.elapsed_seconds >= zs.threshold_seconds:
                        color = COLOR_LOITER_ALERT
                        label = f"ID {t.track_id} LOITERING {zs.elapsed_seconds:.1f}s"
                    else:
                        color = COLOR_LOITER_WARN
                        label = f"ID {t.track_id} {zs.elapsed_seconds:.1f}s in zone"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(annotated, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

        return annotated