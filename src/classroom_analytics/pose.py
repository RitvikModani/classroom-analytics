"""Pose backend seam.

YOLOv8-pose emits detection + 17 COCO keypoints in one pass, so v1's "pose
backend" is just packaging those into a PoseResult. The abstraction exists so a
sharper head-pose source (e.g. MediaPipe FaceMesh per person-crop) can be
dropped in later without touching detect/classify — COCO-17 has no fine face
landmarks, so gaze stays coarse until then.
"""

from __future__ import annotations

import numpy as np

from .schemas import PoseResult


def pack_keypoints(xy: np.ndarray, conf: np.ndarray) -> PoseResult:
    """xy: (17,2) pixel coords, conf: (17,) -> PoseResult with (17,3)."""
    kp = np.zeros((17, 3), dtype=float)
    n = min(len(xy), 17)
    kp[:n, :2] = xy[:n]
    kp[:n, 2] = conf[:n]
    return PoseResult(kp)
