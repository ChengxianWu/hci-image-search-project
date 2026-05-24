from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "openai/clip-vit-base-patch32"
LOCAL_DIR = Path("models/clip-vit-base-patch32")

FILES = [
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "pytorch_model.bin",
]

def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Downloading CLIP model from: {REPO_ID}")
    print(f"[INFO] Saving files to: {LOCAL_DIR.resolve()}")

    for filename in FILES:
        print(f"\n[INFO] Downloading {filename} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(LOCAL_DIR),
            resume_download=True,
            local_dir_use_symlinks=False,
        )
        print(f"[INFO] Finished: {filename}")

    print("\n[INFO] All required CLIP files have been downloaded.")
    print("[INFO] Local CLIP path: models/clip-vit-base-patch32")

if __name__ == "__main__":
    main()