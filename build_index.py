import json
import os

import numpy as np
from tqdm import tqdm

from clip_utils import CLIPEncoder


IMAGE_DIR = "data/images"
INDEX_DIR = "index"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def collect_images(image_dir):
    """Collect supported image files from the dataset folder."""
    image_paths = []

    for root, _, files in os.walk(image_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, file_name))

    image_paths.sort()
    return image_paths


def infer_category_from_filename(file_name):
    """Infer a simple category from names like apple_001.jpg."""
    name = os.path.splitext(file_name)[0]
    return name.split("_")[0] if "_" in name else "unknown"


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    image_paths = collect_images(IMAGE_DIR)
    if not image_paths:
        raise ValueError(
            f"No images found in {IMAGE_DIR}. Please put images into data/images first."
        )

    print(f"[INFO] Found {len(image_paths)} images.")
    encoder = CLIPEncoder()

    features = []
    metadata = []

    for image_path in tqdm(image_paths, desc="Encoding images"):
        try:
            feature = encoder.encode_image(image_path)
            file_name = os.path.basename(image_path)

            features.append(feature)
            metadata.append(
                {
                    "path": image_path,
                    "file_name": file_name,
                    "category": infer_category_from_filename(file_name),
                }
            )
        except Exception as exc:
            print(f"[WARNING] Failed to process {image_path}: {exc}")

    if not features:
        raise RuntimeError("No image features were generated. Please check the dataset.")

    feature_array = np.array(features).astype("float32")
    np.save(os.path.join(INDEX_DIR, "image_features.npy"), feature_array)

    with open(os.path.join(INDEX_DIR, "image_paths.json"), "w", encoding="utf-8") as f:
        json.dump([item["path"] for item in metadata], f, ensure_ascii=False, indent=2)

    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("[INFO] Index built successfully.")
    print(f"[INFO] Feature shape: {feature_array.shape}")
    print(f"[INFO] Saved to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
