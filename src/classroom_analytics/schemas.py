"""Core data structures and interfaces for the pipeline.

Kept deliberately dependency-light (dataclasses + numpy only) so that the
classifier and its unit tests run without pulling in OpenCV / ultralytics /
FastAPI. Everything downstream speaks these types, so this is the contract
that lets pieces be swapped independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# COCO-17 keypoint layout (the format YOLOv8-pose emits).
# ---------------------------------------------------------------------------
class KP(int, Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Action(str, Enum):
    """Primary per-frame action label.

    OFF_TASK is a *derived*, low-confidence bucket (fused phone + talking) and
    is never assigned as a per-frame primary here — the temporal/aggregate
    layer promotes sustained head-down-not-writing into it. See the design doc.
    """

    HAND_RAISED = "hand_raised"
    HEAD_DOWN = "head_down"
    ATTENTIVE = "attentive"
    OUT_OF_SEAT = "out_of_seat"
    OFF_TASK = "off_task"
    UNKNOWN = "unknown"


# Actions that count as "engaged" for the headline class-attention %.
# Deliberately excludes the low-confidence OFF_TASK bucket.
ENGAGED_ACTIONS = frozenset({Action.ATTENTIVE, Action.HAND_RAISED})


@dataclass
class PoseResult:
    """A single person's 17 keypoints as an (17, 3) array of [x, y, conf]."""

    keypoints: np.ndarray  # shape (17, 3), image-pixel coords + confidence

    def xy(self, kp: "KP") -> np.ndarray:
        return self.keypoints[int(kp), :2]

    def conf(self, kp: "KP") -> float:
        return float(self.keypoints[int(kp), 2])

    def visible(self, kp: "KP", thresh: float) -> bool:
        return self.conf(kp) >= thresh


@dataclass
class Track:
    """One tracked person in one frame.

    frame_w / frame_h are carried so classifiers can reason about position
    within the frame (e.g. standing proxies) without a global handle.
    """

    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    pose: PoseResult
    frame_w: int
    frame_h: int
    det_conf: float = 0.0

    @property
    def bbox_height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def bbox_width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def aspect_ratio(self) -> float:
        w = self.bbox_width
        return self.bbox_height / w if w > 1e-6 else 0.0


@dataclass
class ActionState:
    """Result of classifying one Track in one frame.

    sub_labels holds the honest low-confidence guesses (phone / talking) that
    feed the fused OFF_TASK signal but must never drive the headline number.
    """

    action: Action
    confidence: float = 0.0
    sub_labels: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)  # debug/raw metrics

    @property
    def engaged(self) -> bool:
        return self.action in ENGAGED_ACTIONS


@dataclass
class FrameStats:
    """Per-frame aggregate snapshot pushed to the dashboard."""

    ts: float
    num_people: int
    attention_pct: float
    action_counts: dict
    per_track: dict            # track_id -> smoothed action value (str)
    engagement_scores: dict    # track_id -> rolling engagement 0..1
    alerts: list = field(default_factory=list)


class ActionClassifier(ABC):
    """The swap seam.

    v1 ships a HeuristicClassifier. A trained temporal model (LSTM/GRU over
    pose sequences) can later implement this same interface with zero changes
    downstream. That is the *only* thing the "trained classifier" path needs —
    no training code lives in this repo.
    """

    @abstractmethod
    def classify(self, track: Track) -> ActionState:
        """Map one tracked person (single frame) to an ActionState."""
        raise NotImplementedError
