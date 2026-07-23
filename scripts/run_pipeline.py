"""Headless pipeline preview — the Phase-2 acceptance harness.

Run this BEFORE touching the UI. It opens a window with the annotated feed so
you can physically verify each action fires (raise a hand, put your head down,
stand up), that labels don't flicker, that track IDs stay stable, and — read
the on-frame `fps` — that throughput on the Arc iGPU is acceptable.

    python scripts/run_pipeline.py                 # webcam 0, config defaults
    python scripts/run_pipeline.py --source 1      # a different camera
    python scripts/run_pipeline.py --device cpu    # force CPU degraded mode
    python scripts/run_pipeline.py --headless      # no window, print stats only

Press q in the window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classroom_analytics.config import load_config       # noqa: E402
from classroom_analytics.pipeline import Pipeline         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="camera index or file/RTSP path")
    ap.add_argument("--device", default=None, help="auto | intel:gpu | cpu")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source is not None:
        cfg.source = int(args.source) if args.source.isdigit() else args.source
    if args.device is not None:
        cfg.device = args.device

    pipe = Pipeline(cfg).start()
    print(f"source={cfg.source} device={cfg.device} model={cfg.model} — press q to quit")
    last_print = 0.0
    try:
        for frame, stats in pipe.run():
            now = time.monotonic()
            if now - last_print > 1.0:
                print(f"fps={pipe.fps:5.1f} people={stats.num_people} "
                      f"attention={stats.attention_pct:5.1f}% "
                      f"{stats.action_counts} {stats.alerts}")
                last_print = now
            if not args.headless:
                cv2.imshow("Classroom Analytics — pipeline preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
