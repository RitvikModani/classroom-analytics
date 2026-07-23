"""Rolling aggregate statistics.

Every buffer here is bounded (deque maxlen / pruned dicts) so a multi-hour
session does not grow memory. This layer also does the two things a single
frame cannot: promote *sustained* head-down into the fused OFF_TASK bucket,
and flag *possible* neighbour interaction (the talking proxy) as a
low-confidence alert only — never as a headline number.
"""

from __future__ import annotations

import math
from collections import Counter, deque

from .schemas import Action, ENGAGED_ACTIONS, FrameStats, Track


class Aggregator:
    def __init__(
        self,
        timeline_maxlen: int = 600,
        engagement_window: int = 60,
        off_task_frames: int = 20,
        sleep_frames: int = 90,
        heads_down_alert_pct: float = 30.0,
        low_attention_alert_pct: float = 50.0,
        neighbor_dist_ratio: float = 1.8,
    ):
        self.off_task_frames = off_task_frames
        self.sleep_frames = sleep_frames
        self.heads_down_alert_pct = heads_down_alert_pct
        self.low_attention_alert_pct = low_attention_alert_pct
        self.neighbor_dist_ratio = neighbor_dist_ratio
        self.engagement_window = engagement_window

        self.timeline: deque = deque(maxlen=timeline_maxlen)  # (ts, attention_pct)
        self._engagement: dict[int, deque] = {}               # track_id -> deque[bool]


    def _resolve(self, track_id: int, smoothed: Action, smoother) -> Action:
        """Promote sustained head-down into the fused off-task bucket."""
        if smoothed == Action.HEAD_DOWN and smoother.run_length(track_id) >= self.off_task_frames:
            return Action.OFF_TASK
        return smoothed

    def _update_engagement(self, track_id: int, engaged: bool) -> float:
        buf = self._engagement.get(track_id)
        if buf is None:
            buf = self._engagement[track_id] = deque(maxlen=self.engagement_window)
        buf.append(1 if engaged else 0)
        return sum(buf) / len(buf)

    def _possible_interactions(self, tracks: list[Track], final: dict[int, Action]) -> int:
        """Very low-confidence proximity heuristic for 'talking to neighbour'.
        Two nearby people both in an ambiguous/turned posture. Never trusted as
        fact — surfaced as an alert only. No audio => no real talking detection."""
        count = 0
        cand = [t for t in tracks if final.get(t.track_id) == Action.UNKNOWN]
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                a, b = cand[i], cand[j]
                ax, ay = (a.bbox[0] + a.bbox[2]) / 2, (a.bbox[1] + a.bbox[3]) / 2
                bx, by = (b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2
                dist = math.hypot(ax - bx, ay - by)
                near = self.neighbor_dist_ratio * max(a.bbox_width, b.bbox_width, 1.0)
                if dist < near:
                    count += 1
        return count


    def update(self, ts: float, tracks: list[Track], smoothed: dict, smoother) -> FrameStats:
        active_ids = [t.track_id for t in tracks]

        final: dict[int, Action] = {}
        per_track: dict[int, str] = {}
        counts: Counter = Counter()
        engagement_scores: dict[int, float] = {}

        for t in tracks:
            action = self._resolve(t.track_id, smoothed.get(t.track_id, Action.UNKNOWN), smoother)
            final[t.track_id] = action
            per_track[t.track_id] = action.value
            counts[action.value] += 1
            engaged = action in ENGAGED_ACTIONS
            engagement_scores[t.track_id] = round(self._update_engagement(t.track_id, engaged), 3)

        n = len(tracks)
        engaged_n = sum(counts[a.value] for a in ENGAGED_ACTIONS)
        attention_pct = round(100.0 * engaged_n / n, 1) if n else 0.0
        self.timeline.append((round(ts, 2), attention_pct))

        # prune vanished tracks so memory stays bounded
        for tid in [t for t in self._engagement if t not in set(active_ids)]:
            self._engagement.pop(tid, None)

        alerts = self._build_alerts(n, counts, attention_pct, tracks, final, smoother)

        return FrameStats(
            ts=ts, num_people=n, attention_pct=attention_pct,
            action_counts=dict(counts), per_track=per_track,
            engagement_scores=engagement_scores, alerts=alerts,
        )


    def _build_alerts(self, n, counts, attention_pct, tracks, final, smoother) -> list:
        alerts: list = []
        if n == 0:
            return alerts

        heads_down = counts[Action.HEAD_DOWN.value] + counts[Action.OFF_TASK.value]
        heads_down_pct = 100.0 * heads_down / n
        if heads_down_pct > self.heads_down_alert_pct:
            alerts.append(f"{heads_down_pct:.0f}% heads-down/off-task")
        if attention_pct < self.low_attention_alert_pct:
            alerts.append(f"attention low ({attention_pct:.0f}%)")

        # possible sleepers: head-down sustained well past the off-task threshold
        sleepers = [t.track_id for t in tracks
                    if final.get(t.track_id) in (Action.HEAD_DOWN, Action.OFF_TASK)
                    and smoother.run_length(t.track_id) >= self.sleep_frames]
        if sleepers:
            alerts.append(f"{len(sleepers)} possibly asleep (low confidence)")

        interactions = self._possible_interactions(tracks, final)
        if interactions:
            alerts.append(f"{interactions} possible interaction(s) (low confidence)")
        return alerts
