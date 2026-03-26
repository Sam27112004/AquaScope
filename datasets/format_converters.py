"""
Format converters for dataset unification pipeline.

Converts YOLO annotation formats to COCO format.
"""

from __future__ import annotations

from pathlib import Path


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def yolo_bbox_to_coco(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert YOLO bbox (normalized center+size) to COCO bbox (absolute x,y,w,h).

    Args:
        center_x: Normalized center x coordinate (0-1)
        center_y: Normalized center y coordinate (0-1)
        width: Normalized width (0-1)
        height: Normalized height (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x, y, width, height) in absolute pixel coordinates.
        COCO format uses top-left corner (x, y) and box dimensions.
    """
    x = (center_x - width / 2) * img_width
    y = (center_y - height / 2) * img_height
    w = width * img_width
    h = height * img_height

    # Clamp to image bounds to avoid tiny numeric spillovers from source labels.
    x = _clamp(x, 0.0, float(img_width))
    y = _clamp(y, 0.0, float(img_height))
    w = _clamp(w, 0.0, float(img_width) - x)
    h = _clamp(h, 0.0, float(img_height) - y)
    return (x, y, w, h)


def coco_bbox_to_yolo(
    x: float,
    y: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert COCO bbox (absolute x,y,w,h) to YOLO bbox (normalized cx,cy,w,h)."""
    x = _clamp(x, 0.0, float(img_width))
    y = _clamp(y, 0.0, float(img_height))
    width = _clamp(width, 0.0, float(img_width) - x)
    height = _clamp(height, 0.0, float(img_height) - y)

    center_x = (x + width / 2.0) / img_width
    center_y = (y + height / 2.0) / img_height
    norm_w = width / img_width
    norm_h = height / img_height

    center_x = _clamp(center_x, 0.0, 1.0)
    center_y = _clamp(center_y, 0.0, 1.0)
    norm_w = _clamp(norm_w, 0.0, 1.0)
    norm_h = _clamp(norm_h, 0.0, 1.0)

    return (center_x, center_y, norm_w, norm_h)


def yolo_obb_to_coco(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert YOLO OBB (4 normalized corner points) to axis-aligned COCO bbox.

    Args:
        x1, y1, x2, y2, x3, y3, x4, y4: Normalized corner coordinates (0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x, y, width, height) representing axis-aligned bounding box.
        Calculated as min/max of all corner points.
    """
    # Denormalize coordinates
    xs = [x1 * img_width, x2 * img_width, x3 * img_width, x4 * img_width]
    ys = [y1 * img_height, y2 * img_height, y3 * img_height, y4 * img_height]

    # Calculate axis-aligned bounding rectangle
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_min = _clamp(x_min, 0.0, float(img_width))
    y_min = _clamp(y_min, 0.0, float(img_height))
    x_max = _clamp(x_max, 0.0, float(img_width))
    y_max = _clamp(y_max, 0.0, float(img_height))

    return (x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min))


def parse_yolo_label_file(
    label_path: Path, format_type: str
) -> list[list[float]]:
    """Parse YOLO .txt label file and return list of label lines.

    Args:
        label_path: Path to .txt label file
        format_type: "bbox" for standard YOLO (5 values) or "obb" for OBB (9 values)

    Returns:
        List of labels, each containing [class_id, ...coordinates] as floats
    """
    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            values = [float(v) for v in line.split()]

            # Validate format
            if format_type == "bbox" and len(values) != 5:
                continue  # Skip invalid lines
            elif format_type == "obb" and len(values) != 9:
                continue  # Skip invalid lines

            labels.append(values)

    return labels
