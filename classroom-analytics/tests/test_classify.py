"""Synthetic-keypoint tests for the heuristic classifier.

No camera, no model — we hand-build COCO-17 keypoints for a neutral upright
seated person and mutate one thing per test. Image frame is 640x480, y down.
"""

import numpy as np

from classroom_analytics.classify import HeuristicClassifier
from classroom_analytics.schemas import KP, Action, PoseResult, Track

C = 0.9  # a confident keypoint
clf = HeuristicClassifier()


def make_pose(overrides: dict | None = None) -> PoseResult:
    """Neutral upright, frontal, hands resting, legs hidden (seated at desk)."""
    kp = np.zeros((17, 3), dtype=float)
    kp[KP.NOSE] = (320, 150, C)
    kp[KP.LEFT_EYE] = (330, 140, C)
    kp[KP.RIGHT_EYE] = (310, 140, C)
    kp[KP.LEFT_EAR] = (340, 145, C)
    kp[KP.RIGHT_EAR] = (300, 145, C)
    kp[KP.LEFT_SHOULDER] = (360, 220, C)
    kp[KP.RIGHT_SHOULDER] = (280, 220, C)   # shoulder width 80, mid_y 220
    kp[KP.LEFT_WRIST] = (350, 300, 0.8)     # resting, below shoulders
    kp[KP.RIGHT_WRIST] = (290, 300, 0.8)
    for idx, val in (overrides or {}).items():
        kp[int(idx)] = val
    return PoseResult(kp)


def make_track(pose: PoseResult, bbox=(280, 120, 360, 320)) -> Track:
    return Track(track_id=1, bbox=bbox, pose=pose, frame_w=640, frame_h=480, det_conf=C)


def test_attentive_neutral():
    assert clf.classify(make_track(make_pose())).action == Action.ATTENTIVE


def test_hand_raised_right():
    pose = make_pose({KP.RIGHT_WRIST: (290, 100, C)})  # wrist well above shoulder
    assert clf.classify(make_track(pose)).action == Action.HAND_RAISED


def test_hand_raised_left():
    pose = make_pose({KP.LEFT_WRIST: (350, 90, C)})
    assert clf.classify(make_track(pose)).action == Action.HAND_RAISED


def test_head_down():
    pose = make_pose({KP.NOSE: (320, 210, C), KP.LEFT_EYE: (330, 208, C),
                      KP.RIGHT_EYE: (310, 208, C), KP.LEFT_EAR: (340, 210, C),
                      KP.RIGHT_EAR: (300, 210, C)})
    assert clf.classify(make_track(pose)).action == Action.HEAD_DOWN


def test_standing_out_of_seat():
    pose = make_pose({KP.LEFT_KNEE: (350, 480, 0.8), KP.RIGHT_KNEE: (290, 480, 0.8),
                      KP.LEFT_ANKLE: (350, 600, 0.7), KP.RIGHT_ANKLE: (290, 600, 0.7)})
    track = make_track(pose, bbox=(280, 100, 360, 620))  # aspect ~6.5
    assert clf.classify(track).action == Action.OUT_OF_SEAT


def test_turned_away_is_unknown():
    pose = make_pose({KP.NOSE: (0, 0, 0), KP.LEFT_EYE: (0, 0, 0),
                      KP.RIGHT_EYE: (0, 0, 0)})  # only ears -> not frontal
    assert clf.classify(make_track(pose)).action == Action.UNKNOWN


def test_no_keypoints_is_unknown():
    st = clf.classify(make_track(PoseResult(np.zeros((17, 3), dtype=float))))
    assert st.action == Action.UNKNOWN
    assert st.scores.get("reason") == "no_core_kps"
