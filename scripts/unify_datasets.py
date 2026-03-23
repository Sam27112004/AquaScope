"""
Unify multiple raw datasets into COCO format.

Usage:
    python scripts/unify_datasets.py
    python scripts/unify_datasets.py --validate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.dataset_merger import DatasetMerger
from datasets.dataset_validator import (
    print_dataset_statistics,
    validate_coco_format,
    validate_images_exist,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unify multiple raw datasets into COCO format"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("datasets/raw"),
        help="Directory containing raw datasets (default: datasets/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Output directory (default: datasets)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing output files, don't process",
    )
    return parser.parse_args()


def validate_mode(
    output_annotations_dir: Path, output_images_dir: Path
) -> None:
    """Run validation-only mode on existing outputs.

    Args:
        output_annotations_dir: Directory containing annotation JSON files
        output_images_dir: Directory containing images
    """
    logger.info("\n=== Validation Mode ===\n")

    all_valid = True

    for split in ["train", "val", "test"]:
        ann_file = output_annotations_dir / f"{split}.json"

        if not ann_file.exists():
            logger.warning(f"{split}.json not found, skipping")
            continue

        # Load COCO data
        with open(ann_file, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        logger.info(f"Validating {split}...")

        # Run validations
        errors = validate_coco_format(coco_data)
        errors.extend(validate_images_exist(coco_data, output_images_dir))

        # Report results
        if errors:
            logger.error(f"{split.upper()} validation FAILED ({len(errors)} errors):")
            for err in errors[:10]:  # Show first 10 errors
                logger.error(f"  • {err}")
            if len(errors) > 10:
                logger.error(f"  ... and {len(errors) - 10} more errors")
            all_valid = False
        else:
            logger.info(f"{split.upper()}: ✓ VALID")

        # Print statistics
        print_dataset_statistics(coco_data, split, logger)

    if all_valid:
        logger.info("\n✓ All datasets are valid!")
    else:
        logger.error("\n✗ Some validations failed. See errors above.")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set up paths
    raw_dir = Path(args.raw_dir).resolve()
    output_images_dir = Path(args.output_dir) / "processed" / "images"
    output_annotations_dir = Path(args.output_dir) / "annotations"

    logger.info("=" * 60)
    logger.info("AquaScope Dataset Unification Pipeline")
    logger.info("=" * 60)
    logger.info(f"Raw datasets: {raw_dir}")
    logger.info(f"Output images: {output_images_dir}")
    logger.info(f"Output annotations: {output_annotations_dir}")
    logger.info("")

    # Validation-only mode
    if args.validate_only:
        validate_mode(output_annotations_dir, output_images_dir)
        return

    # Full pipeline mode
    logger.info("Starting dataset unification...\n")

    # Create merger and run pipeline
    merger = DatasetMerger(
        raw_dir=raw_dir,
        output_images_dir=output_images_dir,
        output_annotations_dir=output_annotations_dir,
        logger=logger,
    )

    try:
        results = merger.merge_all()

        # Validate all outputs
        logger.info(f"\n{'=' * 60}")
        logger.info("Validating Merged Datasets")
        logger.info("=" * 60)

        all_valid = True

        for split, path in results.items():
            # Load merged data
            with open(path, "r", encoding="utf-8") as f:
                coco_data = json.load(f)

            logger.info(f"\nValidating {split}...")

            # Run validations
            errors = validate_coco_format(coco_data)
            errors.extend(validate_images_exist(coco_data, output_images_dir))

            # Report results
            if errors:
                logger.error(f"{split.upper()} validation FAILED:")
                for err in errors[:10]:
                    logger.error(f"  • {err}")
                if len(errors) > 10:
                    logger.error(f"  ... and {len(errors) - 10} more errors")
                all_valid = False
            else:
                logger.info(f"{split.upper()}: ✓ VALID")

            # Print statistics
            print_dataset_statistics(coco_data, split, logger)

        # Final summary
        logger.info(f"\n{'=' * 60}")
        if all_valid:
            logger.info("✓ Pipeline completed successfully!")
            logger.info("✓ All datasets validated")
            logger.info("Ready for training with UnderwaterDataset class")
        else:
            logger.error("✗ Validation failed. Check errors above.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
