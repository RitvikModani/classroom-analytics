"""Fetch YOLOv8-pose weights and (optionally) export to OpenVINO for the Arc iGPU.

    python models/download_models.py                 # download yolov8n-pose.pt
    python models/download_models.py --size s        # small model
    python models/download_models.py --openvino      # also export OpenVINO IR

The OpenVINO export produces  yolov8<size>-pose_openvino_model/  — point
config.yaml `model:` at that folder and set `device: intel:gpu` to run on the
Intel Arc integrated GPU.
"""

from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="n", choices=["n", "s", "m"],
                    help="model size: n(ano)=fastest, s(mall), m(edium)")
    ap.add_argument("--openvino", action="store_true", help="also export OpenVINO IR")
    args = ap.parse_args()

    from ultralytics import YOLO

    name = f"yolov8{args.size}-pose.pt"
    print(f"downloading / loading {name} ...")
    model = YOLO(name)  # downloads on first use
    print("weights ready:", name)

    if args.openvino:
        print("exporting OpenVINO IR (for Intel Arc iGPU) ...")
        path = model.export(format="openvino")
        print("OpenVINO model at:", path)
        print("-> set config.yaml model to that folder and device: intel:gpu")


if __name__ == "__main__":
    main()
