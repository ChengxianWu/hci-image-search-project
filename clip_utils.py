import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class LightweightFallbackEncoder:
    """
    Offline fallback encoder.

    It is only used when the real CLIP model is not available.
    This keeps the UI runnable, but retrieval quality is weaker than real CLIP.
    """

    def __init__(self, dim=512):
        self.dim = dim
        print("[INFO] Falling back to a lightweight local encoder.")

    def _normalize(self, vec):
        vec = vec.astype("float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def encode_text(self, text):
        # Use a stable hash instead of Python's built-in hash(), which may vary
        # between processes.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self.dim)
        return self._normalize(vec)

    def encode_image(self, image_path_or_pil):
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        image = image.resize((64, 64))
        arr = np.asarray(image).astype("float32") / 255.0

        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))

        base = np.concatenate([mean, std])
        vec = np.zeros(self.dim, dtype="float32")
        vec[: len(base)] = base

        return self._normalize(vec)


class CLIPEncoder:
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        local_model_dir="models/clip-vit-base-patch32",
        allow_fallback=True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.fallback = None

        self.force_clip = os.environ.get("FORCE_CLIP", "0") == "1"
        local_model_path = Path(local_model_dir)

        print(f"[INFO] Using device: {self.device}")

        try:
            if self._has_local_clip_files(local_model_path):
                print(f"[INFO] Loading real CLIP from local path: {local_model_path}")
                self.model = CLIPModel.from_pretrained(
                    str(local_model_path),
                    local_files_only=True,
                ).to(self.device)

                self.processor = CLIPProcessor.from_pretrained(
                    str(local_model_path),
                    local_files_only=True,
                )

            else:
                if self.force_clip:
                    raise FileNotFoundError(
                        f"Local CLIP model not found in {local_model_path}. "
                        "Please run: python scripts/download_clip.py or "
                        "python scripts/download_clip_direct.py"
                    )

                print(f"[WARNING] Local CLIP model not found: {local_model_path}")
                print(f"[INFO] Trying to load CLIP from Hugging Face: {model_name}")

                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)

            self.model.eval()
            print("[INFO] Real CLIP model loaded successfully.")

        except Exception as e:
            print(f"[WARNING] Failed to load real CLIP: {e}")

            if self.force_clip:
                raise RuntimeError(
                    "FORCE_CLIP=1 is set, so fallback is disabled. "
                    "Please download the real CLIP model first."
                ) from e

            if not allow_fallback:
                raise

            self.fallback = LightweightFallbackEncoder(dim=512)

    def _has_local_clip_files(self, local_model_path):
        if not local_model_path.exists():
            return False

        has_config = (local_model_path / "config.json").exists()
        has_processor = (local_model_path / "preprocessor_config.json").exists()
        has_vocab = (local_model_path / "vocab.json").exists()
        has_merges = (local_model_path / "merges.txt").exists()
        has_weight = (
            (local_model_path / "pytorch_model.bin").exists()
            or (local_model_path / "model.safetensors").exists()
        )

        return has_config and has_processor and has_vocab and has_merges and has_weight

    def _normalize_torch_feature(self, features):
        """
        Convert CLIP outputs to a normalized tensor.

        Normally CLIPModel.get_image_features() / get_text_features() returns a
        torch.Tensor directly. This helper is deliberately defensive: if an
        output object such as BaseModelOutputWithPooling is returned by mistake,
        it extracts pooler_output instead of calling .norm() on the whole object.
        """
        if isinstance(features, torch.Tensor):
            feature_tensor = features
        elif hasattr(features, "pooler_output") and features.pooler_output is not None:
            feature_tensor = features.pooler_output
        elif hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            feature_tensor = features.last_hidden_state[:, 0]
        else:
            raise TypeError(
                f"Unsupported CLIP output type: {type(features)}. "
                "Expected a torch.Tensor or an output object with pooler_output."
            )

        feature_tensor = feature_tensor / feature_tensor.norm(dim=-1, keepdim=True)
        return feature_tensor

    def encode_image(self, image_path_or_pil):
        if self.fallback is not None:
            return self.fallback.encode_image(image_path_or_pil)

        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # Correct CLIP feature extraction API. This should return a Tensor.
            image_features = self.model.get_image_features(**inputs)

        image_features = self._normalize_torch_feature(image_features)

        return image_features.cpu().numpy()[0].astype("float32")

    def encode_text(self, text):
        if self.fallback is not None:
            return self.fallback.encode_text(text)

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            # Correct CLIP feature extraction API. This should return a Tensor.
            text_features = self.model.get_text_features(**inputs)

        text_features = self._normalize_torch_feature(text_features)

        return text_features.cpu().numpy()[0].astype("float32")

    def is_real_clip(self):
        return self.fallback is None
