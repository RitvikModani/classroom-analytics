"""Heuristic action classifier (v1).

Pure numpy + schemas — no camera, no model, no torch. That keeps it unit
testable with synthetic keypoints and makes it the clean reference impl behind
the ActionClassifier interface.

Image coordinates: y increases DOWNWARD. All vertical thresholds are expressed
as a fraction of the person's shoulder width so they are scale invariant
(a kid near the camera and one at the back get judged the same way).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import KP, Action, ActionClassifier, ActionState, PoseResult, Track


@dataclass
class Thresholds:
    kp_conf: float = 0.30          # min keypoint confidence to trust a point
    hand_raise_ratio: float = 0.40  # wrist above shoulder by this * shoulder_width
    head_down_ratio: float = 0.35   # nose-above-shoulder gap below this = head down
    head_up_ratio: float = 0.55     # ...above this = clearly upright
    standing_aspect: float = 2.2    # bbox h/w above this (+ legs visible) = standing
    min_core_conf: float = 0.30     # need shoulders this confident or it's UNKNOWN


def _shoulder_scale(pose: PoseResult, track: Track, t: Thresholds) -> float:
    """Person size reference = shoulder width, with bbox fallback."""
    ls, rs = KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER
    if pose.visible(ls, t.kp_conf) and pose.visible(rs, t.kp_conf):
        sw = float(np.linalg.norm(pose.xy(ls) - pose.xy(rs)))
        if sw > 1e-3:
            return sw
    # Fallback: torso is ~0.4 of the bbox width for a roughly frontal person.
    return max(track.bbox_width * 0.4, 1e-3)


def _head_y(pose: PoseResult, t: Thresholds) -> float | None:
    """Best available head anchor: nose, else eye avg, else ear avg."""
    for group in ([KP.NOSE], [KP.LEFT_EYE, KP.RIGHT_EYE], [KP.LEFT_EAR, KP.RIGHT_EAR]):
        ys = [pose.xy(k)[1] for k in group if pose.visible(k, t.kp_conf)]
        if ys:
            return float(np.mean(ys))
    return None


def _shoulder_mid_y(pose: PoseResult, t: Thresholds) -> float | None:
    ys = [pose.xy(k)[1] for k in (KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER)
          if pose.visible(k, t.kp_conf)]
    return float(np.mean(ys)) if ys else None


def _hand_raised(pose: PoseResult, scale: float, t: Thresholds):
    """Either wrist clearly above its shoulder -> raised. Returns (bool, conf)."""
    best = 0.0
    for wrist, shoulder in ((KP.LEFT_WRIST, KP.LEFT_SHOULDER),
                            (KP.RIGHT_WRIST, KP.RIGHT_SHOULDER)):
        if pose.visible(wrist, t.kp_conf) and pose.visible(shoulder, t.kp_conf):
            above = (pose.xy(shoulder)[1] - pose.xy(wrist)[1]) / scale  # +ve = above
            if above > t.hand_raise_ratio:
                # confidence grows with how far above and how sure the wrist is
                margin = min((above - t.hand_raise_ratio) / t.hand_raise_ratio, 1.0)
                best = max(best, 0.5 * pose.conf(wrist) + 0.5 * margin)
    return best > 0.0, best


def _frontal(pose: PoseResult, t: Thresholds) -> bool:
    """Facing the camera enough to read posture: both shoulders + at least one
    eye/nose. Back-of-head (only ears) or a hard side-turn reads as not frontal."""
    shoulders = (pose.visible(KP.LEFT_SHOULDER, t.kp_conf)
                 and pose.visible(KP.RIGHT_SHOULDER, t.kp_conf))
    face = any(pose.visible(k, t.kp_conf)
               for k in (KP.NOSE, KP.LEFT_EYE, KP.RIGHT_EYE))
    return shoulders and face


def _standing(track: Track, pose: PoseResult, t: Thresholds):
    """Weak proxy: tall bbox AND lower body visible. Seated-at-desk people have
    hidden legs and a short/wide box. Real out-of-seat needs a seat map — see
    the design doc; this is the webcam-era stand-in."""
    legs_visible = any(pose.visible(k, t.kp_conf) for k in
                       (KP.LEFT_KNEE, KP.RIGHT_KNEE, KP.LEFT_ANKLE, KP.RIGHT_ANKLE))
    tall = track.aspect_ratio > t.standing_aspect
    return legs_visible and tall


class HeuristicClassifier(ActionClassifier):
    """v1 geometric classifier. One frame in, one ActionState out.

    Priority order matters: an explicit hand-raise beats everything, standing
    beats posture, head-down beats attentive. OFF_TASK is NOT decided here —
    sub_labels carry the honest low-confidence hint and the aggregate layer
    promotes sustained head-down into the fused off-task bucket.
    """

    def __init__(self, thresholds: Thresholds | None = None):
        self.t = thresholds or Thresholds()

    def classify(self, track: Track) -> ActionState:
        pose, t = track.pose, self.t

        sh_mid_y = _shoulder_mid_y(pose, t)
        head_y = _head_y(pose, t)
        if sh_mid_y is None or head_y is None:
            return ActionState(Action.UNKNOWN, 0.0, scores={"reason": "no_core_kps"})

        scale = _shoulder_scale(pose, track, t)
        head_gap = (sh_mid_y - head_y) / scale  # +ve = head above shoulders (upright)

        raised, raise_conf = _hand_raised(pose, scale, t)
        standing = _standing(track, pose, t)
        head_down = head_gap < t.head_down_ratio
        upright = head_gap > t.head_up_ratio
        frontal = _frontal(pose, t)

        scores = {"head_gap": round(head_gap, 3),
                  "aspect": round(track.aspect_ratio, 3),
                  "scale": round(scale, 2)}

        # --- priority resolution ---
        if raised:
            return ActionState(Action.HAND_RAISED, raise_conf, scores=scores)

        if standing:
            return ActionState(Action.OUT_OF_SEAT, 0.6, scores=scores)

        if head_down:
            # honest hint: head-down could be writing OR phone OR sleeping. We
            # do NOT claim which. Aggregate layer + duration decide off-task.
            depth = min((t.head_down_ratio - head_gap) / max(t.head_down_ratio, 1e-3), 1.0)
            conf = 0.4 + 0.4 * depth
            return ActionState(
                Action.HEAD_DOWN, min(conf, 0.85),
                sub_labels={"head_down_ambiguous": round(depth, 2)},
                scores=scores,
            )

        if upright and frontal:
            return ActionState(Action.ATTENTIVE, 0.6, scores=scores)

        # Upright-ish but turned away, or in the ambiguous middle band.
        return ActionState(Action.UNKNOWN, 0.3,
                           scores={**scores, "reason": "ambiguous_posture"})
