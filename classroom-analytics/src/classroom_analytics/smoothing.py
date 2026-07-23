"""Temporal smoothing of per-track action labels.

Raw per-frame classification flickers (a keypoint drops out for one frame and
the label flips). We keep a short ring buffer per track and majority-vote it.
Buffers for tracks that vanish are evicted so memory is bounded by the number
of people currently on screen, not by session length.
"""

from __future__ import annotations

from collections import Counter, deque

from .schemas import Action


class TemporalSmoother:
    def __init__(self, window: int = 9):
        self.window = window
        self._buf: dict[int, deque] = {}
        self._run: dict[int, tuple] = {}  # track_id -> (Action, consecutive_frames)

    def update(self, track_id: int, action: Action) -> Action:
        buf = self._buf.get(track_id)
        if buf is None:
            buf = self._buf[track_id] = deque(maxlen=self.window)
        buf.append(action)

        smoothed = Counter(buf).most_common(1)[0][0]

        prev_action, prev_n = self._run.get(track_id, (None, 0))
        self._run[track_id] = (smoothed, prev_n + 1 if smoothed == prev_action else 1)
        return smoothed

    def run_length(self, track_id: int) -> int:
        """Consecutive smoothed frames in the current state (for sleep / off-task)."""
        return self._run.get(track_id, (None, 0))[1]

    def prune(self, active_ids) -> None:
        active = set(active_ids)
        for tid in [t for t in self._buf if t not in active]:
            self._buf.pop(tid, None)
            self._run.pop(tid, None)
