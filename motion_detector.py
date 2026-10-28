"""Real-time moving object detection with OpenCV.

Compares each frame against a background model, isolates the regions that
changed, and draws bounding boxes around moving objects. Works with a webcam
or a video file.

Examples:
    python motion_detector.py                       # default webcam
    python motion_detector.py --source video.mp4    # video file
    python motion_detector.py --method mog2 --output result.mp4
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np

BOX_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
NORMAL_COLOR = (255, 255, 255)


@dataclass
class Config:
    method: str = "running-avg"   # "first-frame" | "running-avg" | "mog2"
    width: int = 600              # frames are resized to this width
    blur: int = 21                # Gaussian kernel size (odd)
    threshold: int = 25           # pixel difference considered "changed"
    min_area: int = 500           # ignore contours smaller than this (px^2)
    alpha: float = 0.05           # learning rate for running average


class MotionDetector:
    """Background-subtraction motion detector."""

    def __init__(self, config: Config) -> None:
        self.cfg = config
        self._background: np.ndarray | None = None
        self._mog2 = (
            cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            if config.method == "mog2"
            else None
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.cfg.blur, self.cfg.blur), 0)

    def foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Return a binary mask (255 = moving pixel) for the given frame."""
        if self._mog2 is not None:
            mask = self._mog2.apply(cv2.GaussianBlur(frame, (5, 5), 0))
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            return cv2.dilate(mask, None, iterations=2)

        gray = self._preprocess(frame)
        if self._background is None:
            self._background = gray.astype("float32")
            return np.zeros_like(gray)

        delta = cv2.absdiff(cv2.convertScaleAbs(self._background), gray)
        if self.cfg.method == "running-avg":
            cv2.accumulateWeighted(gray, self._background, self.cfg.alpha)

        _, mask = cv2.threshold(delta, self.cfg.threshold, 255, cv2.THRESH_BINARY)
        return cv2.dilate(mask, None, iterations=2)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return bounding boxes (x, y, w, h) of moving objects."""
        mask = self.foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            cv2.boundingRect(c)
            for c in contours
            if cv2.contourArea(c) >= self.cfg.min_area
        ]


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def annotate(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), BOX_COLOR, 2)
    moving = bool(boxes)
    label = f"Motion detected ({len(boxes)})" if moving else "No motion"
    cv2.putText(frame, f"Status: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, ALERT_COLOR if moving else NORMAL_COLOR, 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detect moving objects in a webcam feed or video file.")
    p.add_argument("--source", default="0",
                   help="camera index (e.g. 0) or path to a video file (default: 0)")
    p.add_argument("--method", choices=["first-frame", "running-avg", "mog2"],
                   default="running-avg", help="background model (default: running-avg)")
    p.add_argument("--min-area", type=int, default=500,
                   help="minimum contour area in pixels to count as motion (default: 500)")
    p.add_argument("--threshold", type=int, default=25,
                   help="pixel difference threshold, 0-255 (default: 25)")
    p.add_argument("--width", type=int, default=600, help="processing width in pixels (default: 600)")
    p.add_argument("--output", help="optional path to save the annotated video (.mp4)")
    p.add_argument("--no-display", action="store_true",
                   help="don't open a preview window (useful on servers)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source {args.source!r}", file=sys.stderr)
        return 1

    detector = MotionDetector(Config(method=args.method, width=args.width,
                                     threshold=args.threshold, min_area=args.min_area))
    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = motion_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize(frame, args.width)
            boxes = detector.detect(frame)
            annotate(frame, boxes)

            frames += 1
            motion_frames += bool(boxes)

            if args.output:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Moving Object Detection  (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frames} frames, motion in {motion_frames}.")
    if args.output:
        print(f"Saved annotated video to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
