"""FastAPI + WebSocket backend.

The pipeline runs on a background thread and updates a single latest-snapshot
(annotated JPEG + stats + attention timeline). WebSocket clients are pushed
that snapshot at a fixed rate, decoupled from inference FPS, so a slow frame
never stalls the socket and a fast one never floods it.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import sys
import threading
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from classroom_analytics.config import load_config          # noqa: E402
from classroom_analytics.pipeline import Pipeline            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"

app = FastAPI(title="Classroom Behavior Analytics")
_state = {"jpeg": None, "stats": None, "timeline": [], "fps": 0.0}
_lock = threading.Lock()
_pipeline: Pipeline | None = None


def _run_pipeline(cfg):
    global _pipeline
    _pipeline = Pipeline(cfg).start()
    q = max(10, min(95, cfg.server.jpeg_quality))
    for frame, stats in _pipeline.run():
        jpeg_b64 = None
        if cfg.server.send_frames:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                jpeg_b64 = base64.b64encode(buf).decode("ascii")
        with _lock:
            _state["jpeg"] = jpeg_b64
            _state["stats"] = dataclasses.asdict(stats)
            _state["fps"] = _pipeline.fps
            _state["timeline"] = list(_pipeline.aggregator.timeline)


@app.on_event("startup")
def _startup():
    cfg = load_config()
    app.state.cfg = cfg
    threading.Thread(target=_run_pipeline, args=(cfg,), daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    if _pipeline:
        _pipeline.stop()


@app.get("/")
def index():
    return FileResponse(str(DASHBOARD / "index.html"))


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    push_hz = 10.0
    try:
        while True:
            with _lock:
                payload = {
                    "jpeg": _state["jpeg"],
                    "stats": _state["stats"],
                    "timeline": _state["timeline"],
                    "fps": _state["fps"],
                }
            await sock.send_json(payload)
            await asyncio.sleep(1.0 / push_hz)
    except WebSocketDisconnect:
        return
    except Exception:
        return


# static assets (app.js, style.css) under /static
app.mount("/static", StaticFiles(directory=str(DASHBOARD)), name="static")
