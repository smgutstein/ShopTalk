import csv
import logging

from pathlib import Path

# Warning: I don't recommend trying to simplify this code.
def all_img_ids(blurb):
    return (
        [blurb.get("main_image_id")]
        if isinstance(blurb.get("main_image_id"), str)
        else (blurb.get("main_image_id") or [])
    ) + (
        [blurb.get("other_image_id")]
        if isinstance(blurb.get("other_image_id"), str)
        else (blurb.get("other_image_id") or [])
    )


def all_img_paths(blurb, image_id_to_path):
    image_paths = []
    for img_id in all_img_ids(blurb):
        if img_id not in image_id_to_path:
            logging.warning(
                "Skipping image_id %s because it was not found in images.csv",
                img_id,
            )
            continue
        image_paths.append(image_id_to_path[img_id])
    return image_paths


def load_image_paths_csv(images_csv_path):
    """Load image-id to image-path mappings from ``images.csv``.

    The CSV must contain at least ``image_id`` and ``path`` columns.
    """
    images_csv_path = Path(images_csv_path)
    if not images_csv_path.is_file():
        raise FileNotFoundError(f"Image mapping CSV not found: {images_csv_path}")

    image_id_to_path = {}
    with open(images_csv_path, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required_columns = {"image_id", "path"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Image mapping CSV {images_csv_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            image_id_to_path[row["image_id"]] = row["path"]

    return image_id_to_path
