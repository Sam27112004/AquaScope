# AquaScope-AI — Underwater Detection Engine
## Persistent Context for Claude Sonnet 4.6

---

## WHAT IS THIS?

AquaScope-AI is the **standalone ML module** of a larger underwater inspection platform.

This repo handles ONLY:
- Dataset engineering
- YOLO model training
- Video/image inference
- Structured output generation
- FastAPI inference endpoints

It will later be consumed by a separate web app (React + Express). Do not build frontend or web UI here.

---

## TECH STACK

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Language     | Python 3.10+                      |
| ML Framework | YOLOv8 (Ultralytics)              |
| API Server   | FastAPI                           |
| Video Tools  | OpenCV                            |
| Data Format  | YOLO (.txt + dataset.yaml) canonical; optional COCO for compatibility |
| Datasets     | 4 unified: TrashCAN, Underwater Trash, Underwater Crack, Concrete Crack |

---

## PIPELINE

```
Video / Images
      ↓
Frame Extraction (OpenCV)
      ↓
YOLO Inference (Ultralytics YOLOv8)
      ↓
Detection Results
      ↓
Structured JSON Output → FastAPI response
```

---

## OUTPUT FORMAT

Every inference call returns:

```json
{
  "frame_id": 42,
  "timestamp": 3.5,
  "detections": [
    {
      "class": "trash",
      "confidence": 0.87,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

---

## DETECTION CLASSES

Target classes (unified from 4 source datasets):
- `trash` — general debris (15.9% train data)
- `plastic` — plastic objects (67.5% train data)
- `fishing_net` — net fragments (0.3% train data)
- `marine_growth` — biological growth on structures (7.3% train data)
- `surface_damage` — cracks and surface damage (8.9% train data)

---

## DEVELOPMENT PHASES

| Phase | Task                                      | Status    |
|-------|-------------------------------------------|-----------|
| 1     | Project structure + environment setup     | ✅ DONE   |
| 2     | Dataset download, merge, YOLO conversion  | ✅ DONE   |
| 3     | Model training + validation               | ⬜ TODO   |
| 4     | Inference pipeline (image + video)        | ⬜ TODO   |
| 5     | FastAPI endpoints                         | ⬜ TODO   |
| 6     | Testing + output validation               | ⬜ TODO   |

> **Phase 2 Complete!** Unified dataset ready: 20,755 images → `datasets/processed/images/{train,val,test}` and `datasets/processed/labels/{train,val,test}`
> **Next:** Phase 3 - Model training with YOLOv8. Ready to configure training pipeline.
> **Update status as you go. Always tell Claude which phase you're in.**

---

## SCOPE BOUNDARIES

### ✅ Build This
- Dataset processing scripts
- YOLO training configs and scripts
- Image and video inference pipeline
- JSON-structured detection output
- FastAPI routes: `/infer/image`, `/infer/video`, `/health`
- Model registry (swap models via config)

### ❌ Do Not Build
- Frontend / UI of any kind
- Drone control or navigation logic
- Real-time hardware interfaces
- True crack detection (mark as approximate only)
- Complex custom architectures — use YOLOv8 as-is

---

## DESIGN RULES

- **Config-driven** — model path, confidence threshold, class list via a single config file (YAML)
- **Modular** — dataset scripts, training, inference each in separate modules
- **No overengineering** — clean, readable Python; avoid unnecessary abstractions
- **Practical ML** — don't claim perfect accuracy; tune thresholds appropriately

---

## HOW TO USE THIS FILE

Paste this at the start of every session, then tell Claude:
1. **Current phase** (e.g., "I'm in Phase 2")
2. **Specific task** (e.g., "write a script to convert TrashCan annotations to YOLO format")
3. **Relevant existing code** if any

That's all Claude needs to give accurate, on-scope output.
