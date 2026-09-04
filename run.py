"""
run.py
CLI entrypoint for the surveillance pipeline.

Usage:
    python run.py --video data/videos/MOT17-04.mp4 --zones config/zones.json --output outputs/
"""

import argparse
import json
import sys
from src.pipeline import SurveillancePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video surveillance pipeline: detection, tracking, zone-based events."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--zones", required=True, help="Path to zones.json config")
    parser.add_argument("--output", default="outputs/", help="Output directory")
    parser.add_argument("--model", default="yolov8n.pt",
                         help="YOLO model weights (yolov8n/s/m.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Minimum confidence for a tracked box to be reported "
                              "(the tracker still associates lower-confidence detections "
                              "internally to survive brief occlusion)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                         help="Inference device")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                         choices=["bytetrack.yaml", "botsort.yaml"],
                         help="Tracker config: bytetrack (fast) or botsort (with re-id)")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Inference resolution. Raise to 960-1280 for high-resolution, "
                              "crowded scenes where distant people are too small to detect "
                              "at 640 (slower).")
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = SurveillancePipeline(
        zones_path=args.zones,
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        tracker_cfg=args.tracker,
        imgsz=args.imgsz,
    )

    try:
        result = pipeline.run(video_path=args.video, output_dir=args.output)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n=== Run Summary ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()