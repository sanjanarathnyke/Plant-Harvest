# Plant-Harvest

AI-driven smart agriculture system using YOLO to detect plant growth stages, health, and optimal harvest timing.

This repository contains code, dataset layout and an example Colab training notebook (plant-predict.ipynb) used to train a YOLO model to classify plant stages: `early`, `growth`, `harvest`, and `maturity`.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Dataset structure](#dataset-structure)
- [Quick start (Colab / Local)](#quick-start-colab--local)
- [Training example](#training-example)
- [Validation & Results](#validation--results)
- [Inference example](#inference-example)
- [Troubleshooting & notes](#troubleshooting--notes)
- [Tips to improve performance](#tips-to-improve-performance)
- [Files produced by training](#files-produced-by-training)
- [License & Contact](#license--contact)

---

## Overview

Plant-Harvest uses Ultralytics YOLO to detect plant growth stages from images. The model helps automate monitoring of plant development and supports decisions such as optimal harvest timing.

The notebook `plant-predict.ipynb` included with the project shows the end-to-end flow used in Colab:
- mounting Google Drive
- preparing dataset
- training YOLO
- validating and running inference

---

## Features

- Train YOLO object detection model on labeled images of plants for stage classification
- Use standard YOLO dataset layout (train/val/test with images and YOLO-format labels)
- Quick Colab-ready example (uses `ultralytics` package)
- Example inference code to run predictions and save visualized outputs

---

## Dataset structure

Expected layout (example from the notebook):

/content/data/
- train/
  - images/
  - labels/  (YOLO txt files: class x_center y_center width height, normalized [0,1])
- valid/
  - images/
  - labels/
- test/
  - images/
  - labels/
- data.yaml

Example `data.yaml` used in the notebook:
```yaml
train: /content/data/train/images
val:   /content/data/valid/images
test:  /content/data/test/images

names:
  0: early
  1: growth
  2: harvest
  3: maturity
```

Label format: YOLO format (one line per object)
`<class_id> <x_center> <y_center> <width> <height>` (all normalized between 0 and 1)

---

## Quick start (Colab)

Prerequisites:
- Google Colab or an environment with Python 3.8+ and required packages.
- ultralytics==8.3.10 was used in the notebook.

Install dependencies (example):
```bash
pip install ultralytics==8.3.10
```

Typical Colab steps shown in the notebook:
1. Mount Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
2. Copy dataset (example where dataset zip lives in Drive):
```bash
cp /content/drive/MyDrive/yolo_dataset/agri.zip /content/
unzip /content/agri.zip -d /content/data
```
3. (Optional) Overwrite or confirm `data.yaml`.

---

## Training example

The exact training snippet used in the included notebook:

```python
from ultralytics import YOLO

# small/compact starter model (the notebook downloads yolo11n.pt)
model = YOLO('yolo11n.pt')   # or use a larger backbone (yolo11s, etc.)

model.train(
    data='/content/data/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=20
)
```

Notes:
- The notebook uses `optimizer=auto` (Ultralytics selects optimizer/lr automatically).
- The training produced `runs/detect/train/weights/best.pt` and `last.pt` by default.

---

## Validation & Results

After training for 50 epochs the notebook validated the best checkpoint and reported these primary metrics:

- Overall:
  - Precision: 0.820
  - Recall: 0.904
  - mAP@50: 0.925
  - mAP@50-95: 0.77

- Per-class mAP (approx):
  - early: 0.95
  - growth: 0.929
  - harvest: 0.984
  - maturity: 0.837

These scores came from the run logs saved under `runs/detect/train`.

---

## Inference example

To run inference on a single image (example from the notebook):

```python
# (after training / or load trained weights)
model = YOLO('runs/detect/train/weights/best.pt')  # or path to the desired checkpoint

results = model.predict(
    source='0209m_JPEG.rf.9689b504a63fa87b9d549ec21cde35df.jpg',
    save=True,
    imgsz=640
)
# Results images are saved under runs/detect/<name>
```

Uploaded examples in Colab were saved to:
`runs/detect/train3` (the notebook saved prediction outputs there).

---

## Troubleshooting & notes

- Corrupt labels detected during training:
  The training log warns about a few images with "non-normalized or out of bounds coordinates". The notebook reported these files:
  - `/content/data/train/images/0506t_JPEG.rf.25a274b4b0034791e7171d7f08436aae.jpg`
  - `/content/data/train/images/0809m_JPEG.rf.d4c35101b6c7953d66e2e3a741a0d294.jpg`
  - `/content/data/train/images/2107t_JPEG.rf.b92de89b7472a4c659482b2133e90a7e.jpg`

  Fix:
  - Open the corresponding `.txt` label files in `train/labels` and ensure all values are normalized between `0.0` and `1.0`.
  - Confirm label lines follow YOLO format: `class x_center y_center width height`.
  - Remove or correct labels where width/height/centers are out of bounds.

- `pin_memory` warning:
  - When running on CPU-only (as in some Colab runtimes), you may see a PyTorch warning about `pin_memory`. It's a benign information message and does not affect training on CPU.

---

## Tips to improve performance

- Use a larger YOLO backbone (yolo11s, yolo11m, or official yolov8/yolov5 models) if compute allows.
- Increase dataset size and class balance across `early`, `growth`, `harvest`, `maturity`.
- Fix and clean labels; remove out-of-bounds or incorrect annotations.
- Use stronger augmentation or longer training (monitor overfitting).
- Consider transfer learning from a model pretrained on a larger dataset or fine-tune backbone layers.
- Use mixed-precision GPU training (if GPU available) to accelerate and scale batch size.

---

## Files produced by training

- runs/detect/train/weights/best.pt — best checkpoint (optimizer stripped)
- runs/detect/train/weights/last.pt — last checkpoint (optimizer stripped)
- runs/detect/train/labels.jpg — plotted label distribution
- runs/detect/train/*.csv, *.json — (depending on options) metrics and logs

---

## License & Contact

- License: MIT (choose a license file and adjust here if you prefer a different one)
- Author / Maintainer: (replace with your name) Sanjana Rathnayake / sanjanarathnyke

If you want, I can:
- create a Git commit that adds this README to your repository (I will need the repository path and permission), or
- help prepare a PR with further improvements (model card, detailed experiments, or export scripts).
