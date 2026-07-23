"""Configuration loading: config.yaml for tuning, .env for secrets.

Nothing about the camera source, thresholds, model, or device is hardcoded in
the pipeline — it all flows from here. Secrets (an RTSP url with user:pass) are
read from the environment / .env and never live in the committed YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields as _fields
from pathlib import Path

import yaml

from .classify import Thresholds


@dataclass
class ServerCfg:
    host: str = "127.0.0.1"
    port: int = 8000
    jpeg_quality: int = 70
    send_frames: bool = True     # push annotated JPEG; False = overlays only


@dataclass
class AggCfg:
    timeline_maxlen: int = 600
    engagement_window: int = 60
    off_task_frames: int = 20
    sleep_frames: int = 90
    heads_down_alert_pct: float = 30.0
    low_attention_alert_pct: float = 50.0
    neighbor_dist_ratio: float = 1.8


@dataclass
class AppConfig:
    source: object = 0            # 0 = default webcam; str path/RTSP; overridable by env
    device: str = "auto"          # auto | intel:gpu | dml | cpu
    model: str = "yolov8n-pose.pt"
    imgsz: int = 640
    det_conf: float = 0.30
    frame_skip: int = 0           # process every (skip+1)-th frame in degraded mode
    queue_size: int = 4           # bounded capture queue
    reconnect_delay: float = 2.0
    smoothing_window: int = 9
    thresholds: Thresholds = field(default_factory=Thresholds)
    aggregate: AggCfg = field(default_factory=AggCfg)
    server: ServerCfg = field(default_factory=ServerCfg)


def _apply(dc, data: dict):
    """Overlay a dict onto a dataclass instance, field by field (shallow-typed)."""
    known = {f.name for f in _fields(dc)}
    for k, v in (data or {}).items():
        if k in known and not isinstance(getattr(dc, k), (Thresholds, AggCfg, ServerCfg)):
            setattr(dc, k, v)
    return dc


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (avoids a hard python-dotenv dependency)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def load_config(config_path: str = "config/config.yaml", env_path: str = ".env") -> AppConfig:
    _load_dotenv(Path(env_path))
    raw: dict = {}
    p = Path(config_path)
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    cfg = AppConfig()
    _apply(cfg, raw)
    if raw.get("thresholds"):
        _apply(cfg.thresholds, raw["thresholds"])
    if raw.get("aggregate"):
        _apply(cfg.aggregate, raw["aggregate"])
    if raw.get("server"):
        _apply(cfg.server, raw["server"])

    # --- secrets / env overrides win over the file ---
    if os.environ.get("RTSP_URL"):
        cfg.source = os.environ["RTSP_URL"]
    if os.environ.get("CLASSROOM_SOURCE"):
        s = os.environ["CLASSROOM_SOURCE"]
        cfg.source = int(s) if s.isdigit() else s
    if os.environ.get("CLASSROOM_DEVICE"):
        cfg.device = os.environ["CLASSROOM_DEVICE"]
    return cfg
