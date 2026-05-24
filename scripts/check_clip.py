from transformers import CLIPModel, CLIPProcessor

MODEL_DIR = "models/clip-vit-base-patch32"

print("[INFO] Loading CLIP model from local directory...")
model = CLIPModel.from_pretrained(MODEL_DIR, local_files_only=True)
print("[INFO] Model loaded successfully.")

print("[INFO] Loading CLIP processor from local directory...")
processor = CLIPProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
print("[INFO] Processor loaded successfully.")

print("[INFO] Real CLIP is ready.")