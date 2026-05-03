"""Build a multimodal FAISS vector database for ShopTalk products.

This script reads product blurbs and an image-id-to-file-path mapping, embeds each
product with Meta ImageBind text and vision encoders, combines the two embeddings,
and writes a FAISS inner-product index plus a FAISS-row-to-product-id mapping.

The intended pipeline is:

1. Load product metadata from ``--product_blurbs``.
2. Load image filename mappings from ``--images_csv``.
3. Filter products down to those with usable image files.
4. Batch text/image inputs through ImageBind.
5. Average normalized text and image embeddings for each product.
6. Normalize the final product embeddings and store them in FAISS.
7. Save generated artifacts under ``artifacts/vector_db`` by default.

Only lightweight validation and bookkeeping live in helper functions. The expensive
model inference path is intentionally concentrated in ``load_multimodal_embeddings``.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")  # Suppress noisy third-party warnings during long embedding runs.

import faiss
import pickle
import torch
from tqdm import tqdm

from imagebind import data
from imagebind.models.imagebind_model import imagebind_huge, ModalityType


# Keep generated binary artifacts out of the repo root by default.
DEFAULT_VECTOR_DB_DIR = Path("artifacts/vector_db")
DEFAULT_FAISS_INDEX_OUTPUT = DEFAULT_VECTOR_DB_DIR / "faiss_index.bin"
DEFAULT_INDEX_MAPPING_OUTPUT = DEFAULT_VECTOR_DB_DIR / "index_to_product_id.pkl"

logger = logging.getLogger(__name__)


def normalize(vectors):
    """L2-normalize a 2D torch tensor row-by-row.

    ImageBind text and vision embeddings are compared with dot products. After
    L2 normalization, dot product search with ``faiss.IndexFlatIP`` is equivalent
    to cosine-similarity search.

    Args:
        vectors: Tensor shaped ``(num_vectors, embedding_dim)``.

    Returns:
        Tensor with the same shape, with each row scaled to unit norm.

    Raises:
        ValueError: If any row has zero norm, since that row cannot be normalized.
    """
    norms = torch.linalg.norm(vectors, axis=1, keepdims=True)
    if torch.any(norms == 0):
        raise ValueError("Cannot normalize embeddings with zero vector norm.")
    return vectors / norms


def validate_input_paths(product_blurbs, image_root, images_csv):
    """Validate input paths before starting expensive model work.

    Failing fast here is intentional. Discovering a bad path after ImageBind has
    already loaded or after several batches have run wastes time and makes errors
    harder to diagnose.
    """
    product_blurbs = Path(product_blurbs)
    image_root = Path(image_root)
    images_csv = Path(images_csv)

    if not product_blurbs.is_file():
        raise FileNotFoundError(f"Product blurbs file not found: {product_blurbs}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root directory not found: {image_root}")
    if not images_csv.is_file():
        raise FileNotFoundError(f"Image mapping CSV not found: {images_csv}")


def validate_args(args):
    """Validate parsed command-line arguments."""
    if args.batch_size <= 0:
        raise ValueError(f"--batch_size must be positive; got {args.batch_size}")

    validate_input_paths(args.product_blurbs, args.image_root, args.images_csv)


def load_img_mappings(csv_path):
    """Load Amazon image IDs to local/server image paths from ``images.csv``.

    The returned paths are relative to ``--image_root``. The CSV is expected to
    contain at least these columns:

    - ``image_id``: Amazon image identifier used in product blurbs.
    - ``path``: relative image path under the image root directory.
    """
    img_id_to_server_filename = {}
    csv_path = Path(csv_path)

    with open(csv_path, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        required_columns = {"image_id", "path"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Image mapping CSV {csv_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            img_id_to_server_filename[row["image_id"]] = row["path"]

    return img_id_to_server_filename


def load_imagebind_model(device):
    """Load the pretrained ImageBind model once and move it to ``device``.

    This must stay outside the batch loop. Recreating ``imagebind_huge`` per batch
    was the main performance bug in the original script.
    """
    logger.info("Loading ImageBind model...")
    ibind_model = imagebind_huge(pretrained=True)
    ibind_model.eval()
    ibind_model.to(device)
    return ibind_model


def build_product_description(blurb):
    """Create the text string embedded for a product.

    The current text representation is deliberately simple: category plus product
    type. Missing fields are given stable fallback values so a malformed product
    record does not crash the whole vector DB build.
    """
    feature_fields = blurb.get("feature_fields") or {}
    categories = feature_fields.get("categories", "Uncategorized")
    product_type = feature_fields.get("product_type", "Unknown")
    return f"{categories}/{product_type}"


def collect_products_with_images(
    blurbs_by_id,
    img_id_to_server_filename,
    img_root,
    skip_missing_images=False,
):
    """Filter product blurbs down to products with usable image files.

    A product is kept only when:

    1. Its ``main_image_id`` appears in the image mapping CSV.
    2. The mapped image file exists under ``img_root``.
    3. The product ID has not already been emitted.

    By default, missing referenced image files are treated as data integrity
    errors. ``--skip_missing_images`` switches this to warning-and-skip behavior.
    """
    img_root = Path(img_root)
    id_blurb_pairs = []
    seen_product_ids = set()
    missing_image_paths = []

    for product_id, blurb in blurbs_by_id.items():
        if product_id in seen_product_ids:
            logger.debug(f"Duplicate product_id found while preparing embeddings: {product_id}")
            continue

        img_id = blurb.get("main_image_id")
        img_filename = img_id_to_server_filename.get(img_id)
        if not img_filename:
            # Product has no mapped image. It cannot receive a multimodal embedding.
            continue

        image_path = img_root / img_filename
        if not image_path.is_file():
            if skip_missing_images:
                logger.warning(f"Skipping product_id {product_id}; missing image file: {image_path}")
                continue
            missing_image_paths.append((product_id, image_path))
            continue

        id_blurb_pairs.append((product_id, blurb))
        seen_product_ids.add(product_id)

    if missing_image_paths:
        preview = "\n".join(
            f"  product_id={product_id}: {image_path}"
            for product_id, image_path in missing_image_paths[:10]
        )
        remaining_count = len(missing_image_paths) - 10
        if remaining_count > 0:
            preview += f"\n  ... and {remaining_count} more missing image files"
        raise FileNotFoundError(
            "Missing image files referenced by product blurbs/images.csv. "
            "Use --skip_missing_images to skip these products.\n"
            f"{preview}"
        )

    return id_blurb_pairs


def build_image_paths(batch_pairs, img_root, img_id_to_server_filename):
    """Build image file paths for one batch of product blurbs.

    ``collect_products_with_images`` already checked that these files exist, so
    this function only translates product blurbs into the ordered path list
    expected by ImageBind's vision preprocessing helper.
    """
    img_root = Path(img_root)
    return [
        str(img_root / img_id_to_server_filename[blurb.get("main_image_id")])
        for _, blurb in batch_pairs
    ]


def load_multimodal_embeddings(
    blurbs_by_id,
    img_root,
    images_csv,
    device,
    batch_size=128,
    skip_missing_images=False,
):
    """Generate one combined text+image embedding per eligible product.

    Text and image embeddings are independently L2-normalized, then averaged.
    The final stack is normalized again in ``create_vectordb`` before being added
    to the FAISS inner-product index.
    """
    img_root = Path(img_root)

    logger.info(f"Loading (Amazon image id -> server filename) mappings from {images_csv}...")
    img_id_to_server_filename = load_img_mappings(images_csv)
    logger.info(f"Total images in img_id_to_server_filename: {len(img_id_to_server_filename)}")

    logger.info("Preparing text and image data...")
    id_blurb_pairs = collect_products_with_images(
        blurbs_by_id,
        img_id_to_server_filename,
        img_root,
        skip_missing_images=skip_missing_images,
    )

    logger.info(f"Filtered down to {len(id_blurb_pairs)} products with valid image IDs.")

    if not id_blurb_pairs:
        raise ValueError("No products with valid image IDs found.")

    descriptions = [build_product_description(blurb) for _, blurb in id_blurb_pairs]

    multimodal_embeddings = []

    # Expensive model setup happens once. The batch loop below only prepares inputs
    # and runs inference against this already-loaded model.
    ibind_model = load_imagebind_model(device)

    num_batches = (len(descriptions) + batch_size - 1) // batch_size

    with tqdm(total=len(descriptions), desc="Processing all batches") as pbar:
        for batch_idx in range(num_batches):
            logger.debug(f"Processing batch {batch_idx + 1}/{num_batches}...")

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(descriptions))

            # Keep these two lists aligned: index i in batch_descriptions must
            # describe the same product as index i in batch_img_load_paths.
            batch_descriptions = descriptions[start_idx:end_idx]
            batch_pairs = id_blurb_pairs[start_idx:end_idx]

            batch_img_load_paths = build_image_paths(
                batch_pairs,
                img_root,
                img_id_to_server_filename,
            )

            logger.debug(f"Number of images to load for this batch: {len(batch_img_load_paths)}")

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(batch_descriptions, device),
                ModalityType.VISION: data.load_and_transform_vision_data(
                    tqdm(batch_img_load_paths, desc="Loading images", leave=False),
                    device,
                ),
            }

            logger.debug(f"Text data shape: {inputs[ModalityType.TEXT].shape}")
            if inputs[ModalityType.VISION] is not None:
                logger.debug(f"Vision data shape before model: {inputs[ModalityType.VISION].shape}")
                logger.debug(f"Vision data type: {inputs[ModalityType.VISION].dtype}")
            else:
                logger.debug("No vision data for this batch")

            logger.debug("Running model inference for batch...")
            with torch.no_grad():
                embeddings = ibind_model(inputs)

            text_embeddings = normalize(embeddings[ModalityType.TEXT])
            vision_embeddings = normalize(embeddings[ModalityType.VISION])

            logger.debug(f"Text embeddings shape: {text_embeddings.shape}")
            logger.debug(f"Vision embeddings shape: {vision_embeddings.shape}")

            if len(text_embeddings) != len(vision_embeddings):
                raise ValueError(
                    f"Mismatched text/image embedding counts: "
                    f"{len(text_embeddings)} text vs {len(vision_embeddings)} vision"
                )

            # The text and image batches were built in the same product order, so
            # the embeddings at position idx correspond to the same product_id.
            for idx, (product_id, _) in enumerate(batch_pairs):
                text_emb = text_embeddings[idx]
                vision_emb = vision_embeddings[idx]
                multimodal_embeddings.append((product_id, (text_emb + vision_emb) / 2))
                logger.debug(f"Adding product_id {product_id} multimodal embedding")

            pbar.update(end_idx - start_idx)

    return multimodal_embeddings


def create_vectordb(
    blurbs_by_id,
    img_root,
    images_csv,
    device,
    batch_size,
    skip_missing_images=False,
):
    """Create a FAISS index and row-to-product-id mapping.

    The FAISS index stores normalized multimodal product embeddings. The separate
    ``index_to_product_id`` dictionary is required because FAISS only returns row
    numbers; the application needs to translate those rows back to product IDs.
    """
    multimodal_embeddings_by_id = load_multimodal_embeddings(
        blurbs_by_id,
        img_root,
        images_csv,
        device,
        batch_size=batch_size,
        skip_missing_images=skip_missing_images,
    )

    logger.info("All multimodal embeddings created")

    multimodal_embeddings = []
    index_to_product_id = {}

    logger.info(f"Total multimodal embeddings: {len(multimodal_embeddings_by_id)}")

    for index, (product_id, embedding) in enumerate(multimodal_embeddings_by_id):
        multimodal_embeddings.append(embedding)
        index_to_product_id[index] = product_id

    if not multimodal_embeddings:
        raise ValueError("No embeddings found in multimodal_embeddings")

    logger.info(f"Embedding shape: {multimodal_embeddings[0].shape}")
    embedding_dim = multimodal_embeddings[0].shape[0]
    logger.info(f"Total embeddings to add to FAISS index: {len(multimodal_embeddings)}")

    # Normalize the averaged multimodal embeddings before inner-product indexing
    # so FAISS scores behave like cosine similarities.
    normalized_embeddings = normalize(torch.vstack(multimodal_embeddings).to(dtype=torch.float32))

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(normalized_embeddings.cpu().numpy())

    return index, index_to_product_id


def save_vectordb(index, index_to_product_id, faiss_index_output, index_mapping_output):
    """Persist the FAISS index and product-id mapping to disk."""
    faiss_index_output = Path(faiss_index_output)
    index_mapping_output = Path(index_mapping_output)

    # Parent directories are created here instead of in main() so callers can reuse
    # this helper safely with arbitrary output locations.
    faiss_index_output.parent.mkdir(parents=True, exist_ok=True)
    index_mapping_output.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(faiss_index_output))
    logger.info(f"FAISS index saved to {faiss_index_output}")

    with open(index_mapping_output, "wb") as file:
        pickle.dump(index_to_product_id, file)
    logger.info(f"Index mapping saved to {index_mapping_output}")


def parse_args():
    """Parse command-line arguments for vector DB generation."""
    parser = argparse.ArgumentParser(
        description="Generate a multimodal FAISS vector database for ShopTalk products."
    )
    parser.add_argument("--product_blurbs", type=str, default="EDA/product_blurbs/combined_blurb_dict.json")
    parser.add_argument("--image_root", type=str, default="server/static/images")
    parser.add_argument("--images_csv", type=str, default="images.csv")
    parser.add_argument("--faiss_index_output", type=str, default=str(DEFAULT_FAISS_INDEX_OUTPUT))
    parser.add_argument("--index_mapping_output", type=str, default=str(DEFAULT_INDEX_MAPPING_OUTPUT))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--skip_missing_images", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main(args):
    """Run the full vector DB build from parsed command-line arguments."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
        datefmt="%H:%M:%S",
    )

    validate_args(args)

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")
    logger.info(f"Generating VectorDB from images in {args.image_root}")
    logger.info(f"Using image mapping CSV: {args.images_csv}")
    logger.info(f"FAISS index output: {args.faiss_index_output}")
    logger.info(f"Index mapping output: {args.index_mapping_output}")
    logger.info(f"Skip missing images: {args.skip_missing_images}")

    logger.info(f"Loading data from {args.product_blurbs}...")
    with open(args.product_blurbs, "r", encoding="utf-8") as f:
        blurbs_by_id = json.load(f)

    index, index_to_product_id = create_vectordb(
        blurbs_by_id,
        args.image_root,
        args.images_csv,
        device,
        args.batch_size,
        skip_missing_images=args.skip_missing_images,
    )

    save_vectordb(
        index,
        index_to_product_id,
        args.faiss_index_output,
        args.index_mapping_output,
    )

    logger.info("VectorDB saved")


if __name__ == "__main__":
    main(parse_args())
