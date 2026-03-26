"""
Validation utilities for merged COCO datasets.

Validates format integrity and reports statistics.
"""

from __future__ import annotations

from pathlib import Path


def validate_yolo_split(
    images_dir: Path,
    labels_dir: Path,
    num_classes: int,
) -> list[str]:
    """Validate one YOLO split directory pair and return errors."""
    errors: list[str] = []
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label file for image: {image_path.name}")
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for line_idx, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"{label_path.name}:{line_idx} has {len(parts)} values (expected 5)")
                continue

            try:
                class_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                errors.append(f"{label_path.name}:{line_idx} contains non-numeric values")
                continue

            if class_id < 0 or class_id >= num_classes:
                errors.append(f"{label_path.name}:{line_idx} invalid class id {class_id}")

            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                errors.append(f"{label_path.name}:{line_idx} center out of range")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                errors.append(f"{label_path.name}:{line_idx} width/height out of range")

    return errors


def print_yolo_split_statistics(images_dir: Path, labels_dir: Path, split: str, logger) -> None:
    """Print summary statistics for one YOLO split."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    image_paths = sorted(
        p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    label_paths = sorted(labels_dir.glob("*.txt"))

    total_boxes = 0
    empty_labels = 0
    for label_path in label_paths:
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        total_boxes += len(lines)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Statistics for {split.upper()}")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total images: {len(image_paths):,}")
    logger.info(f"Total label files: {len(label_paths):,}")
    logger.info(f"Total objects: {total_boxes:,}")
    logger.info(f"Images without objects: {empty_labels:,}")
    if image_paths:
        logger.info(f"Avg objects per image: {total_boxes / len(image_paths):.2f}")


def validate_coco_format(coco_data: dict) -> list[str]:
    """Validate COCO JSON structure and return list of errors.

    Args:
        coco_data: Loaded COCO JSON dict

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check required keys
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in coco_data:
            errors.append(f"Missing required key: {key}")
            continue

    # Validate images
    image_ids = []
    for img in coco_data.get("images", []):
        if "id" not in img:
            errors.append(f"Image missing id field: {img}")
            continue

        img_id = img["id"]
        if img_id in image_ids:
            errors.append(f"Duplicate image ID: {img_id}")
        image_ids.append(img_id)

        # Check required fields
        for field in ["file_name", "width", "height"]:
            if field not in img:
                errors.append(f"Image {img_id} missing required field: {field}")

    # Convert to set for fast lookup
    image_id_set = set(image_ids)

    # Validate annotations
    ann_ids = []
    for ann in coco_data.get("annotations", []):
        if "id" not in ann:
            errors.append("Annotation missing id field")
            continue

        ann_id = ann["id"]
        if ann_id in ann_ids:
            errors.append(f"Duplicate annotation ID: {ann_id}")
        ann_ids.append(ann_id)

        # Check image_id reference
        if "image_id" not in ann:
            errors.append(f"Annotation {ann_id} missing image_id")
        elif ann["image_id"] not in image_id_set:
            errors.append(
                f"Annotation {ann_id} references unknown image_id {ann['image_id']}"
            )

        # Check bbox format
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"Annotation {ann_id} has invalid bbox: {bbox}")
        else:
            # Check that bbox values are numeric
            if not all(isinstance(v, (int, float)) for v in bbox):
                errors.append(f"Annotation {ann_id} bbox contains non-numeric values")

        # Check category_id exists
        if "category_id" not in ann:
            errors.append(f"Annotation {ann_id} missing category_id")

    # Validate categories
    cat_ids = set()
    for cat in coco_data.get("categories", []):
        if "id" not in cat or "name" not in cat:
            errors.append(f"Category missing required fields: {cat}")
            continue
        cat_ids.add(cat["id"])

    # Check all annotations reference valid categories
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id and cat_id not in cat_ids:
            errors.append(
                f"Annotation {ann.get('id')} references unknown category_id {cat_id}"
            )

    return errors


def validate_images_exist(coco_data: dict, images_dir: Path) -> list[str]:
    """Check that all referenced images exist on disk.

    Args:
        coco_data: Loaded COCO JSON dict
        images_dir: Directory containing images

    Returns:
        List of error messages for missing images
    """
    errors = []
    images_dir = Path(images_dir)

    for img in coco_data.get("images", []):
        img_path = images_dir / img["file_name"]
        if not img_path.exists():
            errors.append(f"Image file not found: {img_path}")

    return errors


def print_dataset_statistics(coco_data: dict, split: str, logger) -> None:
    """Print comprehensive dataset statistics.

    Args:
        coco_data: Loaded COCO JSON dict
        split: Split name for display
        logger: Logger instance
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Statistics for {split.upper()}")
    logger.info("=" * 60)

    num_images = len(coco_data.get("images", []))
    num_annotations = len(coco_data.get("annotations", []))

    logger.info(f"Total images: {num_images:,}")
    logger.info(f"Total annotations: {num_annotations:,}")

    if num_images > 0:
        logger.info(f"Avg annotations per image: {num_annotations / num_images:.2f}")

    # Annotations per category
    cat_counts = {}
    for ann in coco_data.get("annotations", []):
        cat_id = ann["category_id"]
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

    # Get category names
    cat_names = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    logger.info("\nAnnotations by class:")
    for cat_id in sorted(cat_counts.keys()):
        count = cat_counts[cat_id]
        name = cat_names.get(cat_id, f"unknown_{cat_id}")
        percentage = (count / num_annotations * 100) if num_annotations > 0 else 0
        logger.info(f"  {name:15} (id={cat_id}): {count:6,} ({percentage:5.1f}%)")

    # Images by source dataset
    source_counts = {}
    for img in coco_data.get("images", []):
        source = img.get("source_dataset", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1

    logger.info("\nImages by source dataset:")
    for source in sorted(source_counts.keys()):
        count = source_counts[source]
        percentage = (count / num_images * 100) if num_images > 0 else 0
        logger.info(f"  {source:20}: {count:6,} ({percentage:5.1f}%)")
