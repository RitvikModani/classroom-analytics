"""Person detection + pose + tracking in one pass (YOLOv8-pose + ByteTrack).

ultralytics gives boxes, 17 keypoints, and persistent track IDs from a single
model.track() call — cheaper than detect -> crop -> separate pose, which
matters on an Intel Arc iGPU with no CUDA headroom.

Device: 'intel:gpu' runs the OpenVINO-exported model on the Arc iGPU; 'cpu' is
the portable fallback; 'auto' lets ultralytics choose (CPU unless CUDA exists,
which here it won't). 'dml' isn't natively supported by ultralytics inference
and falls back to CPU with a warning — see the README for the ORT-DML path.
"""

from __future__ import annotations

import logging

from .pose import pack_keypoints
from .schemas import Track

log = logging.getLogger(__name__)


class PersonTracker:
    def __init__(self, cfg):
        from ultralytics import YOLO  # lazy: only needed when actually detecting

        self.cfg = cfg
        self.model = YOLO(cfg.model)
        self.device = self._resolve_device(cfg.device)
        log.info("YOLO pose model=%s device=%s", cfg.model, self.device)

    @staticmethod
    def _resolve_device(device: str):
        if device in ("auto", "", None):
            return None  # ultralytics default
        if device == "dml":
            log.warning("device 'dml' not supported by ultralytics inference; using CPU")
            return "cpu"
        return device  # 'intel:gpu' | 'cpu' | explicit


    def track(self, frame) -> list[Track]:
        h, w = frame.shape[:2]
        kwargs = dict(persist=True, tracker="bytetrack.yaml", imgsz=self.cfg.imgsz,
                      conf=self.cfg.det_conf, classes=[0], verbose=False)
        if self.device is not None:
            kwargs["device"] = self.device

        res = self.model.track(frame, **kwargs)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None or res.keypoints is None:
            return []  # nothing tracked this frame

        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()        # (n, 17, 2)
        kconf = res.keypoints.conf                   # (n, 17) or None
        kconf = kconf.cpu().numpy() if kconf is not None else None

        tracks: list[Track] = []
        for i, tid in enumerate(ids):
            conf_row = kconf[i] if kconf is not None else [1.0] * len(kxy[i])
            tracks.append(Track(
                track_id=int(tid),
                bbox=tuple(float(v) for v in xyxy[i]),
                pose=pack_keypoints(kxy[i], conf_row),
                frame_w=w, frame_h=h,
                det_conf=float(confs[i]),
            ))
        return tracks
