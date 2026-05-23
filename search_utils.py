import json
import os
import zipfile

import numpy as np


INDEX_DIR = "index"
OUTPUT_DIR = "outputs"


def load_index():
    """Load image vectors, image paths, and metadata."""
    features_path = os.path.join(INDEX_DIR, "image_features.npy")
    paths_path = os.path.join(INDEX_DIR, "image_paths.json")
    metadata_path = os.path.join(INDEX_DIR, "metadata.json")

    if not os.path.exists(features_path):
        raise FileNotFoundError("image_features.npy not found. Please run python build_index.py first.")
    if not os.path.exists(paths_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Index metadata not found. Please run python build_index.py first.")

    image_features = np.load(features_path)

    with open(paths_path, "r", encoding="utf-8") as f:
        image_paths = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return image_features, image_paths, metadata


def search_by_vector(query_vector, image_features, image_paths, metadata, top_k=5):
    """Search by cosine similarity. Vectors are normalized, so dot product is enough."""
    if len(image_features) == 0:
        return []

    query_vector = query_vector.astype("float32")
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    query_vector = query_vector / query_norm
    top_k = max(1, min(int(top_k), len(image_features)))

    scores = image_features @ query_vector
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        item = metadata[idx]
        results.append(
            {
                "rank": rank,
                "path": image_paths[idx],
                "score": float(scores[idx]),
                "file_name": item.get("file_name", os.path.basename(image_paths[idx])),
                "category": item.get("category", "unknown"),
            }
        )

    return results


def results_to_gallery(results):
    """Convert results to the format accepted by gr.Gallery."""
    gallery_items = []

    for item in results:
        caption = (
            f"Rank {item['rank']} | "
            f"Score: {item['score']:.4f} | "
            f"{item['category']} | "
            f"{item['file_name']}"
        )
        gallery_items.append((item["path"], caption))

    return gallery_items


def build_result_summary(results):
    """Build a compact text overview for the ranked results."""
    if not results:
        return "No results found."

    best = results[0]
    return (
        f"Total results: {len(results)}\n"
        f"Best match: {best['file_name']}\n"
        f"Category: {best['category']}\n"
        f"Best similarity score: {best['score']:.4f}"
    )


def create_results_zip(results):
    """Package ranked result images into a ZIP file for download."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    zip_path = os.path.join(OUTPUT_DIR, "search_results.zip")

    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in results:
            image_path = item["path"]
            file_name = os.path.basename(image_path)
            archive_name = f"rank_{item['rank']:02d}_score_{item['score']:.4f}_{file_name}"
            zipf.write(image_path, archive_name)

    return zip_path
