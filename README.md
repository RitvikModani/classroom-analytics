# Classroom Behavior Analytics (CV)

Real-time, **privacy-preserving** engagement analytics for a classroom camera.
No facial recognition, no names — every person gets a per-session tracker ID
(ByteTrack) so behaviour is counted per-seat without identifying anyone.

Pose landmarks (YOLOv8-pose) + geometric heuristics classify actions per person
per frame, smooth them over time, and aggregate into live class-wide stats on a
dashboard.

## What it detects — and how much to trust each

| Action | Confidence | Honest caveat |
|---|---|---|
| **Hand raised** | High | Wrist clearly above shoulder. Reliable. |
| **Out of seat / standing** | High | Tall bbox + visible legs. (Real per-seat detection needs a seat map — deferred until a fixed classroom camera.) |
| **Head down** | Medium | Reported as *head down*, **not** "asleep". Sleep is only guessed after head-down is sustained. |
| **Attentive** | Medium | This is a **posture proxy** (head up, facing front), not true attention. The desk hides the hands — we can't see writing. |
| **Off-task** *(phone + talking, fused)* | Low | Phone is below the desk (invisible); "talking" has no audio. These are **fused into one low-confidence bucket** and shown as *possible* signals only. They never drive the headline attention %. |

The headline **class-attention %** is computed only from the reliable signals
(attentive + hand-raised). This is deliberate — see `docs`/design notes. Don't
let the demo imply phone/talking detection is precise; it isn't, and the code
is built to be honest about that.

## Architecture

```
capture.py  -> detect.py -> classify.py -> smoothing.py -> aggregate.py -> server.py -> dashboard/
(threaded     (YOLOv8-pose  (heuristics    (per-track      (rolling         (FastAPI +    (live UI)
 grabber,      + ByteTrack,   per frame)     majority vote)  bounded stats)   WebSocket)
 bounded Q)    track IDs)
```

Every module has one job and talks through `schemas.py` types, so pieces can be
swapped or tested in isolation.


## Setup

```bash
python -m venv .venv && .venv\Scripts\activate    # Windows PowerShell
pip install -r requirements.txt

cp config/config.example.yaml config/config.yaml   # then tune
cp .env.example .env                               # only if using RTSP/secrets

python models/download_models.py                   # fetch yolov8n-pose weights
```

## Run

**1. Prove the pipeline first (do this before caring about the UI):**

```bash
python scripts/run_pipeline.py
```

A window opens with the annotated feed. Raise a hand, put your head down, stand
up — confirm each label fires, doesn't flicker, track IDs stay stable, and read
the on-frame **fps**. This is the go/no-go gate.

**2. Run the dashboard:**

```bash
uvicorn server.server:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

Unit tests (no camera/model needed):

```bash
pytest -q
```

## Compute — Intel Arc integrated GPU (no CUDA)

This machine has an **Intel Arc iGPU**, not an NVIDIA card, so there's no CUDA.

- **OpenVINO (recommended):** `python models/download_models.py --openvino`
  exports `yolov8n-pose_openvino_model/`. Point `config.yaml`
  `model:` at that folder and set `device: intel:gpu`.
- **CPU (portable fallback):** `device: cpu`. Works anywhere, slower.
- **DirectML path (optional):** any DX12 GPU incl. Arc via
  `onnxruntime-directml` + an ONNX export. Not wired into ultralytics inference
  by default — manual path, uncomment the dep in `requirements.txt`.

**FPS is a target to measure, not a promise.** On an iGPU expect real-time for
a few people on a webcam; a full ~25-person classroom will likely need the nano
model + lower `imgsz` + `frame_skip`. Measure at the pipeline gate first.

**Degraded mode:** set `device: cpu`, `model: yolov8n-pose.pt`, and
`frame_skip: 1` (or higher) to keep video smooth while inferring less often.


## Configuration

Nothing is hardcoded. `config/config.yaml` holds all tuning (source, device,
model, thresholds, alert levels, buffer sizes). **Secrets go in `.env`**, never
in the YAML:

- `RTSP_URL=rtsp://user:pass@host/stream` — overrides `source` (this is a
  credential; that's why it's separated).
- `CLASSROOM_SOURCE`, `CLASSROOM_DEVICE` — env overrides for quick switching.

Env values always win over the file.

## Swapping in a trained classifier later

The heuristics are one implementation of the `ActionClassifier` interface in
[`src/classroom_analytics/schemas.py`](src/classroom_analytics/schemas.py):

```python
class ActionClassifier(ABC):
    @abstractmethod
    def classify(self, track: Track) -> ActionState: ...
```

To replace heuristics with a trained temporal model (e.g. an LSTM/GRU over pose
sequences), implement this interface and swap it into the pipeline:

1. Write `TrainedClassifier(ActionClassifier)` — buffer each track's recent
   `PoseResult`s, run your model, return an `ActionState`.
2. In `pipeline.py`, replace `HeuristicClassifier(...)` with your class.

Nothing else changes — detection, tracking, smoothing, aggregation, server, and
dashboard all speak the same `ActionState`. **No training code lives in this
repo** by design; this is only the seam that makes adding it painless.

## Guarantees

- **Bounded memory:** every history buffer is a capped `deque`; the capture
  queue drops oldest under backpressure; vanished tracks are evicted. A
  multi-hour session doesn't grow.
- **Resilient camera:** unplug/replug (or an RTSP drop) triggers a
  reconnect loop — the pipeline and server keep running, showing "SOURCE DOWN".
- **Privacy:** anonymous per-session IDs only. No enrollment, no identity.
