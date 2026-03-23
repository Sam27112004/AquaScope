# Aquascope — AI Underwater Inspection Analysis System

An end-to-end machine learning platform for underwater drone inspection footage.
Supports multi-model object detection, segmentation, classification, video inference,
experiment tracking, and a REST API backend.

## Project Structure

```
aquascope/
├── config/           # YAML configuration files (training, inference, base)
├── datasets/         # Dataset loaders, augmentations, preprocessing
├── models/           # Pluggable model definitions (detection, segmentation, classification)
├── training/         # Training loop, losses, metrics, callbacks
├── inference/        # Video & image inference pipelines, postprocessing
├── experiments/      # Logs, checkpoints, and result artefacts
├── api/              # FastAPI web backend with routes and schemas
├── utils/            # Shared utilities (logging, visualisation, file I/O)
└── scripts/          # CLI entry points for training, evaluation, and inference
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python scripts/train.py --config config/training_config.yaml

# Run video inference
python scripts/infer_video.py --source path/to/video.mp4 --config config/inference_config.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pt

# Start the API server
uvicorn api.app:app --reload
```

## Adding a New Model

1. Create a class in the appropriate `models/<task>/` sub-package that inherits from `models.base_model.BaseModel`.
2. Implement `build()`, `forward()`, and `load_checkpoint()`.
3. Register the model slug in `config/base_config.yaml` under `models.registry`.
4. The training and inference pipelines will pick it up automatically.

## Requirements

- Python 3.10+
- PyTorch 2.x
- See `requirements.txt` for the full dependency list.
