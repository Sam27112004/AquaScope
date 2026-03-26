"""
Unify multiple raw datasets into YOLO format.

Usage:
    python scripts/unify_datasets.py
    python scripts/unify_datasets.py --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.dataset_merger import DatasetMerger
from datasets.dataset_validator import (
    print_yolo_split_statistics,
    validate_yolo_split,
)
from datasets.class_mapper import ClassMapper
from utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unify multiple raw datasets into YOLO format"
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
        default=Path("datasets/processed"),
        help="Output processed directory (default: datasets/processed)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing output files, don't process",
    )
    return parser.parse_args()


def validate_mode(output_processed_dir: Path) -> None:
    """Run validation-only mode on existing YOLO outputs."""
    logger.info("\n=== Validation Mode ===\n")

    all_valid = True
    num_classes = len(ClassMapper.get_yolo_class_names())

    for split in ["train", "val", "test"]:
        images_dir = output_processed_dir / "images" / split
        labels_dir = output_processed_dir / "labels" / split

        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"{split} split not found under processed/images+labels, skipping")
            continue

        logger.info(f"Validating {split}...")

        errors = validate_yolo_split(images_dir, labels_dir, num_classes=num_classes)

        if errors:
            logger.error(f"{split.upper()} validation FAILED ({len(errors)} errors):")
            for err in errors[:10]:  # Show first 10 errors
                logger.error(f"  • {err}")
            if len(errors) > 10:
                logger.error(f"  ... and {len(errors) - 10} more errors")
            all_valid = False
        else:
            logger.info(f"{split.upper()}: ✓ VALID")

        print_yolo_split_statistics(images_dir, labels_dir, split, logger)

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
    output_processed_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("AquaScope Dataset Unification Pipeline")
    logger.info("=" * 60)
    logger.info(f"Raw datasets: {raw_dir}")
    logger.info(f"Output processed dir: {output_processed_dir}")
    logger.info("")

    # Validation-only mode
    if args.validate_only:
        validate_mode(output_processed_dir)
        return

    # Full pipeline mode
    logger.info("Starting dataset unification...\n")

    # Create merger and run pipeline
    merger = DatasetMerger(
        raw_dir=raw_dir,
        output_processed_dir=output_processed_dir,
        logger=logger,
    )

    try:
        merger.merge_all()

        # Validate all outputs
        logger.info(f"\n{'=' * 60}")
        logger.info("Validating Unified YOLO Dataset")
        logger.info("=" * 60)

        validate_mode(output_processed_dir)

        # Final summary
        logger.info(f"\n{'=' * 60}")
        logger.info("✓ Pipeline completed successfully!")
        logger.info("✓ Unified YOLO dataset validated")
        logger.info("Ready for training from datasets/processed/dataset.yaml")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
