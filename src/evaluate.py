"""
evaluate.py
Computes MOTA/MOTP by comparing our tracker's output against MOT17 ground truth.

Usage:
    python -m src.evaluate --video data/MOT17/train/MOT17-04-FRCNN/img1 \
                            --gt data/MOT17/train/MOT17-04-FRCNN/gt/gt.txt
"""

import argparse
import cv2
import numpy as np

# motmetrics still calls np.asfarray, which numpy removed in 2.0. This repo
# pins numpy<2.0 (see requirements.txt) so this is normally a no-op, but the
# patch keeps evaluate.py working standalone if numpy is ever upgraded.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

import motmetrics as mm
from pathlib import Path

from src.tracker import PersonTracker


def load_gt(gt_path):
    """Returns {frame_number: [(track_id, x, y, w, h), ...]}"""
    gt = {}
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            frame, tid, x, y, w, h, conf = parts[:7]
            frame, tid = int(frame), int(tid)
            conf = float(conf)
            if conf == 0:  # MOT17 marks ignored boxes with conf=0
                continue
            x, y, w, h = float(x), float(y), float(w), float(h)
            gt.setdefault(frame, []).append((tid, x, y, w, h))
    return gt


def run_on_image_sequence(img_dir: str, tracker: PersonTracker, num_frames: int = None):
    """Returns {frame_number: [(track_id, x, y, w, h), ...]} from our tracker."""
    img_dir = Path(img_dir)
    frame_files = sorted(img_dir.glob("*.jpg"))
    if num_frames:
        frame_files = frame_files[:num_frames]

    results = {}
    for i, fpath in enumerate(frame_files, start=1):  # MOT frames are 1-indexed
        frame = cv2.imread(str(fpath))
        tracks = tracker.track(frame)
        results[i] = [(t.track_id, t.bbox[0], t.bbox[1],
                        t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1]) for t in tracks]
    return results


def evaluate(gt, pred, name="sequence"):
    acc = mm.MOTAccumulator(auto_id=True)

    all_frames = sorted(set(gt.keys()) | set(pred.keys()))
    for frame in all_frames:
        gt_boxes = gt.get(frame, [])
        pred_boxes = pred.get(frame, [])

        gt_ids = [b[0] for b in gt_boxes]
        pred_ids = [b[0] for b in pred_boxes]

        gt_xywh = np.array([b[1:] for b in gt_boxes]) if gt_boxes else np.empty((0, 4))
        pred_xywh = np.array([b[1:] for b in pred_boxes]) if pred_boxes else np.empty((0, 4))

        dist_matrix = mm.distances.iou_matrix(gt_xywh, pred_xywh, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dist_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "motp", "num_switches",
                                        "num_false_positives", "num_misses"],
                          name=name)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to MOT17 img1 directory")
    parser.add_argument("--gt", required=True, help="Path to gt.txt")
    parser.add_argument("--num_frames", type=int, default=None,
                         help="Limit frames for a faster test run")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Minimum confidence for a tracked box to be reported")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         choices=["bytetrack.yaml", "botsort.yaml"])
    args = parser.parse_args()

    print("Loading ground truth...")
    gt = load_gt(args.gt)

    print(f"Running tracker on image sequence (conf={args.conf}, imgsz={args.imgsz}, "
          f"tracker={args.tracker})...")
    tracker = PersonTracker(model_path=args.model, conf_threshold=args.conf,
                             device=args.device, tracker_cfg=args.tracker,
                             imgsz=args.imgsz)
    pred = run_on_image_sequence(args.video, tracker, args.num_frames)

    # only evaluate on the frames we actually predicted on, so an
    # unprocessed tail of the video doesn't get counted as all-misses
    if args.num_frames:
        gt = {f: boxes for f, boxes in gt.items() if f in pred}

    print("Computing MOTA/MOTP...")
    # gt.txt lives at <sequence>/gt/gt.txt, so its grandparent is the sequence name
    seq_name = Path(args.gt).resolve().parent.parent.name
    summary = evaluate(gt, pred, name=seq_name)
    print(summary)


if __name__ == "__main__":
    main()