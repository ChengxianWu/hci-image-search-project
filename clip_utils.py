import hashlib
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """Load CLIP when available, with a local fallback for offline demos."""
        self.mode = os.environ.get("IMAGE_SEARCH_ENCODER", "auto").lower()
        self.model_name = os.environ.get("CLIP_MODEL_NAME", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.backend = "fallback"

        if self.mode in {"fallback", "simple"}:
            print("[INFO] Using local fallback encoder.")
            self.model = None
            self.processor = None
            return

        try:
            print(f"[INFO] Using device: {self.device}")
            print(f"[INFO] Loading CLIP model: {self.model_name}")
            local_only = self.mode == "auto"
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            ).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )
            self.model.eval()
            self.backend = "clip"
            print("[INFO] CLIP model loaded.")
        except Exception as exc:
            if self.mode == "clip":
                raise
            print(f"[WARNING] CLIP model is not available locally: {exc}")
            print("[INFO] Falling back to a lightweight local encoder.")
            self.model = None
            self.processor = None

    def encode_image(self, image_path_or_pil):
        """Encode an image path or PIL image into a normalized CLIP vector."""
        source_name = None
        if isinstance(image_path_or_pil, str):
            source_name = os.path.basename(image_path_or_pil).lower()
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        if self.backend == "fallback":
            return self._encode_image_fallback(image, source_name)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0]

    def encode_text(self, text):
        """Encode text into a normalized CLIP vector."""
        if self.backend == "fallback":
            return self._encode_text_fallback(text)

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0]

    def _encode_image_fallback(self, image, source_name=None):
        resized = image.resize((64, 64))
        pixels = np.asarray(resized).astype("float32") / 255.0
        mean_rgb = pixels.mean(axis=(0, 1))
        std_rgb = pixels.std(axis=(0, 1))

        vector = np.zeros(512, dtype="float32")
        vector[0:3] = mean_rgb
        vector[3:6] = std_rgb

        hist_features = []
        for channel in range(3):
            hist, _ = np.histogram(pixels[:, :, channel], bins=16, range=(0.0, 1.0))
            hist_features.extend(hist.astype("float32") / pixels.size)
        vector[6:54] = np.array(hist_features, dtype="float32")
        self._add_keyword_features(vector, source_name or "")

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def _encode_text_fallback(self, text):
        text = (text or "").lower()
        vector = np.zeros(512, dtype="float32")

        matched = False
        for keyword, (rgb, _) in self._keyword_map().items():
            if keyword in text:
                vector[0:3] += np.array(rgb, dtype="float32")
                matched = True
        matched = self._add_keyword_features(vector, text) or matched

        if not matched:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            for i, byte in enumerate(digest):
                vector[6 + i] = byte / 255.0
            vector[0:3] = np.array([0.6, 0.6, 0.6], dtype="float32")

        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def _keyword_map(self):
        return {
            "apple": ((0.82, 0.18, 0.20), 80),
            "red": ((0.85, 0.16, 0.16), 70),
            "banana": ((0.92, 0.78, 0.18), 100),
            "yellow": ((0.92, 0.78, 0.18), 72),
            "milk": ((0.92, 0.96, 1.00), 120),
            "bottle": ((0.86, 0.92, 1.00), 110),
            "orange": ((0.93, 0.50, 0.14), 140),
            "bread": ((0.70, 0.46, 0.25), 160),
            "tomato": ((0.82, 0.18, 0.18), 180),
            "carrot": ((0.90, 0.43, 0.13), 200),
            "grape": ((0.45, 0.28, 0.61), 220),
            "purple": ((0.45, 0.28, 0.61), 230),
        }

    def _add_keyword_features(self, vector, text):
        matched = False
        for keyword, (_, offset) in self._keyword_map().items():
            if keyword in text:
                vector[offset] = 3.0
                matched = True
        return matched
