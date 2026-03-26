"""
Class mapping utilities for dataset unification.

Maps source dataset classes to unified target classes for AquaScope-AI.
"""

from __future__ import annotations


class ClassMapper:
    """Maps source dataset classes to unified target classes.

    Target classes:
        1: trash - general debris
        2: plastic - plastic objects
        3: fishing_net - net fragments
        4: marine_growth - biological growth on structures
        5: surface_damage - approximate damage markers
    """

    # Target classes (COCO uses 1-indexed category IDs)
    TARGET_CLASSES = {
        1: "trash",
        2: "plastic",
        3: "fishing_net",
        4: "marine_growth",
        5: "surface_damage",
    }

    # TrashCAN class name → target class ID
    # None means exclude from dataset
    TRASHCAN_MAP = {
        # Marine organisms → marine_growth (4)
        "plant": 4,
        "animal_fish": 4,
        "animal_starfish": 4,
        "animal_crab": 4,
        "animal_eel": 4,
        "animal_shells": 4,
        "animal_etc": 4,
        # General trash → trash (1)
        "trash_bag": 1,
        "trash_bottle": 1,
        "trash_clothing": 1,
        "trash_pipe": 1,
        "trash_can": 1,
        "trash_cup": 1,
        "trash_container": 1,
        "trash_snack_wrapper": 1,
        "trash_branch": 1,
        "trash_wreckage": 1,
        "trash_tarp": 1,
        "trash_unknown_instance": 1,
        "trash_rope": 1,
        # Fishing net → fishing_net (3)
        "trash_net": 3,
        # Exclude ROV
        "rov": None,
    }

    # Simple dataset class mappings
    SIMPLE_MAP = {
        "trash_plastic": 2,  # plastic
        "crack": 5,  # surface_damage
        "Concrete-Crack": 5,  # surface_damage
    }

    @classmethod
    def map_trashcan_class(cls, class_name: str) -> int | None:
        """Map TrashCAN class name to target class ID.

        Args:
            class_name: Original TrashCAN class name

        Returns:
            Target class ID (1-5), or None to exclude this class
        """
        return cls.TRASHCAN_MAP.get(class_name)

    @classmethod
    def map_simple_class(cls, class_name: str) -> int | None:
        """Map simple dataset class name to target class ID.

        Args:
            class_name: Original class name from Roboflow datasets

        Returns:
            Target class ID (1-5), or None if unmapped
        """
        return cls.SIMPLE_MAP.get(class_name)

    @classmethod
    def get_coco_categories(cls) -> list[dict]:
        """Return COCO categories list for target classes.

        Returns:
            List of category dicts in COCO format
        """
        return [
            {"id": class_id, "name": name, "supercategory": "object"}
            for class_id, name in cls.TARGET_CLASSES.items()
        ]

    @classmethod
    def coco_id_to_yolo_id(cls, coco_class_id: int) -> int:
        """Convert 1-indexed COCO class ID to 0-indexed YOLO class ID."""
        if coco_class_id not in cls.TARGET_CLASSES:
            raise ValueError(f"Unknown COCO class ID: {coco_class_id}")
        return coco_class_id - 1

    @classmethod
    def yolo_id_to_coco_id(cls, yolo_class_id: int) -> int:
        """Convert 0-indexed YOLO class ID to 1-indexed COCO class ID."""
        coco_class_id = yolo_class_id + 1
        if coco_class_id not in cls.TARGET_CLASSES:
            raise ValueError(f"Unknown YOLO class ID: {yolo_class_id}")
        return coco_class_id

    @classmethod
    def get_yolo_class_names(cls) -> list[str]:
        """Return YOLO class names ordered by 0-indexed class ID."""
        return [cls.TARGET_CLASSES[i] for i in sorted(cls.TARGET_CLASSES.keys())]
