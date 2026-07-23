"""End-to-end pipeline: capture -> detect/track -> classify -> smooth -> aggregate.

Exposed as a generator yielding (annotated_frame, FrameStats). The runner
script and the server both consume it the same way; neither knows about the
internals. A measured FPS is drawn on-frame so the Phase-2 gate can read real
throughput on the Arc iGPU.
"""

from __future__ import annotations

import time
from collections import deque

import cv2

from .aggregate import Aggregator
from .capture import FrameGrabber
from .classify import HeuristicClassifier
from .schemas import Action, FrameStats
from .smoothing import TemporalSmoother

COLORS = {
    Action.HAND_RAISED: (0, 200, 0),
    Action.ATTENTIVE: (0, 160, 0),
    Action.HEAD_DOWN: (0, 165, 255),
    Action.OFF_TASK: (0, 0, 235),
    Action.OUT_OF_SEAT: (255, 120, 0),
    Action.UNKNOWN: (150, 150, 150),
}


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.classifier = HeuristicClassifier(cfg.thresholds)
        self.smoother = TemporalSmoother(cfg.smoothing_window)
        a = cfg.aggregate
        self.aggregator = Aggregator(
            timeline_maxlen=a.timeline_maxlen, engagement_window=a.engagement_window,
            off_task_frames=a.off_task_frames, sleep_frames=a.sleep_frames,
            heads_down_alert_pct=a.heads_down_alert_pct,
            low_attention_alert_pct=a.low_attention_alert_pct,
            neighbor_dist_ratio=a.neighbor_dist_ratio,
        )
        self.grabber: FrameGrabber | None = None
        self.tracker = None
        self._fps = deque(maxlen=30)

    def start(self):
        from .detect import PersonTracker  # lazy: pulls in ultralytics
        self.tracker = PersonTracker(self.cfg)
        self.grabber = FrameGrabber(self.cfg.source, self.cfg.queue_size,
                                    self.cfg.reconnect_delay).start()
        return self

    def stop(self):
        if self.grabber:
            self.grabber.stop()


    def run(self):
        """Yield (annotated_frame, FrameStats) forever until stop()."""
        assert self.grabber and self.tracker, "call start() first"
        skip = max(0, self.cfg.frame_skip)
        i = 0
        last_tracks, last_stats = [], self._empty_stats()

        while True:
            frame = self.grabber.read(timeout=1.0)
            if frame is None:
                yield self._source_down_frame(), last_stats
                continue

            i += 1
            if skip and (i % (skip + 1)) != 0:
                # degraded mode: reuse last inference, keep video smooth
                yield self._annotate(frame.copy(), last_tracks, last_stats), last_stats
                continue

            t0 = time.monotonic()
            tracks = self.tracker.track(frame)
            smoothed = {}
            for tr in tracks:
                raw = self.classifier.classify(tr)
                smoothed[tr.track_id] = self.smoother.update(tr.track_id, raw.action)
            self.smoother.prune([t.track_id for t in tracks])
            stats = self.aggregator.update(t0, tracks, smoothed, self.smoother)

            self._fps.append(1.0 / max(time.monotonic() - t0, 1e-6))
            last_tracks, last_stats = tracks, stats
            yield self._annotate(frame, tracks, stats), stats


    @property
    def fps(self) -> float:
        return round(sum(self._fps) / len(self._fps), 1) if self._fps else 0.0

    def _empty_stats(self) -> FrameStats:
        return FrameStats(ts=0.0, num_people=0, attention_pct=0.0,
                          action_counts={}, per_track={}, engagement_scores={})

    def _source_down_frame(self):
        import numpy as np
        f = np.zeros((480, 640, 3), dtype="uint8")
        cv2.putText(f, "SOURCE DOWN - reconnecting...", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 235), 2)
        return f

    def _annotate(self, frame, tracks, stats: FrameStats):
        for tr in tracks:
            action = Action(stats.per_track.get(tr.track_id, Action.UNKNOWN.value))
            color = COLORS.get(action, (150, 150, 150))
            x1, y1, x2, y2 = (int(v) for v in tr.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            score = stats.engagement_scores.get(tr.track_id, 0.0)
            label = f"#{tr.track_id} {action.value} [{score:.0%}]"
            cv2.putText(frame, label, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        head = f"attention {stats.attention_pct:.0f}%  people {stats.num_people}  fps {self.fps}"
        cv2.putText(frame, head, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if stats.alerts:
            cv2.putText(frame, " | ".join(stats.alerts), (8, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 235), 2)
        return frame
