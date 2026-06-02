"""Helpers for resolving product image paths for the Gradio UI."""

import logging
from pathlib import Path

from .shoptalk_paths import STATIC_IMAGES_DIR


def gradio_image_path(image_path):
    """Return a local image path usable by Gradio, or None if missing."""
    normalized_image_path = str(image_path).replace("\\", "/").lstrip("/")

    candidates = [
        Path(image_path),
        STATIC_IMAGES_DIR / normalized_image_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    logging.warning(
        "Image not found: %s; checked paths: %s",
        image_path,
        [str(candidate.resolve()) for candidate in candidates],
    )
    return None


def chosen_product_image_paths(chosen_product):
    """Return local image paths for the chosen product's image gallery."""
    if not chosen_product:
        return []

    image_paths = []
    for product_image_path in chosen_product.get("image_paths") or []:
        resolved_path = gradio_image_path(product_image_path)
        if resolved_path is not None:
            image_paths.append(resolved_path)

    return image_paths


def top_product_image_paths(result, max_images=12):
    """Return local image paths for retrieved products in diagnostics order."""
    diagnostics = result.get("diagnostics") or {}
    top_products = diagnostics.get("top_products") or []

    image_paths = []
    for product in top_products:
        for product_image_path in product.get("image_paths") or []:
            resolved_path = gradio_image_path(product_image_path)
            if resolved_path is None:
                continue
            if resolved_path not in image_paths:
                image_paths.append(resolved_path)
            if len(image_paths) >= max_images:
                return image_paths

    return image_paths