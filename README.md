# HCI Lab 3: Image Search System

This project implements an image search interface for Human-Computer Interaction Lab 3.

The system supports both **Text-to-Image Search** and **Image-to-Image Search**. It uses CLIP embeddings to encode text queries and image queries into a shared vector space, then retrieves the most similar images by cosine similarity.

When the real CLIP model is unavailable due to network limitations, the project can automatically fall back to a lightweight offline encoder so that the complete user interface and search workflow can still run.

---

## Features

- Text-to-Image Search
- Image-to-Image Search
- Ranked image retrieval by cosine similarity
- Top-K result control
- Query preview
- Result overview
- Download retrieved images as a ZIP file
- Real CLIP mode and offline fallback mode
- Gradio-based interactive user interface

---

## Technical Stack

- Python
- Gradio
- PyTorch
- Hugging Face Transformers
- CLIP
- Pillow
- NumPy

---

## Project Structure

```text
hci-image-search-project/
├── app.py
├── build_index.py
├── clip_utils.py
├── search_utils.py
├── requirements.txt
├── README.md
├── data/
│   └── images/
├── index/
│   ├── image_features.npy
│   ├── image_paths.json
│   └── metadata.json
├── outputs/
├── models/
│   └── clip-vit-base-patch32/
└── scripts/
    ├── download_clip.py
    ├── download_clip_direct.py
    └── check_clip.py
```

Note: models/, .cache/, and outputs/ are local runtime directories and should not be uploaded to GitHub.

## Environment

Python 3.10 is recommended.

The project can also run on newer Python versions if all dependencies are installed successfully.

## Installation

### Option 1: Use an existing virtual environment

Activate your local environment first.

For Windows PowerShell:

```powershell
cd D:\image_search
.\.venv\Scripts\activate
```

For Windows CMD:

```cmd
cd /d D:\image_search
.\.venv\Scripts\activate
```

Then install dependencies:

```bash
python -m pip install -r requirements.txt
```

If the default source is slow, use a mirror:

```bash
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

### Option 2: Create a conda environment

```bash
conda create -n hci_lab3 python=3.10 -y
conda activate hci_lab3
pip install -r requirements.txt
```

## Prepare Dataset

Put image files into:

```text
data/images
```

Supported image formats:

```text
jpg, jpeg, png, bmp, webp
```

Example:

```text
data/images/apple_001.jpg
data/images/apple_002.jpg
data/images/banana_001.jpg
data/images/milk_001.jpg
```

## Encoder Modes

This project supports two encoder modes.

### 1. Real CLIP Mode

This mode uses the real CLIP model:

```text
openai/clip-vit-base-patch32
```

It provides better semantic retrieval quality and should be used for the final experiment whenever possible.

### 2. Fallback Encoder Mode

If the real CLIP model cannot be downloaded or loaded, the project can use a lightweight offline fallback encoder.

The fallback encoder is only used to keep the interface and workflow runnable. Its retrieval quality is weaker than real CLIP.

## Download Real CLIP Model

The real CLIP model should be downloaded into:

```text
models/clip-vit-base-patch32
```

### Use Hugging Face Hub script

PowerShell:

```powershell
cd D:\image_search
.\.venv\Scripts\activate

$env:HF_HOME="D:\image_search\.cache\huggingface"
$env:HF_ENDPOINT="https://hf-mirror.com"

python scripts\download_clip.py
```

CMD:

```cmd
cd /d D:\image_search
.\.venv\Scripts\activate

set HF_HOME=D:\image_search\.cache\huggingface
set HF_ENDPOINT=https://hf-mirror.com

python scripts\download_clip.py
```

## Check Local CLIP Model

After downloading, check whether the real CLIP model can be loaded locally:

```bash
python scripts\check_clip.py
```

Successful output should include:

```text
[INFO] Real CLIP is ready.
```

The local model directory should contain files such as:

```text
config.json
preprocessor_config.json
tokenizer_config.json
special_tokens_map.json
vocab.json
merges.txt
pytorch_model.bin
```

or:

```text
model.safetensors
```

## Build Image Index

The image index must be built before running the application.

### Build with real CLIP

PowerShell:

```powershell
cd D:\image_search
.\.venv\Scripts\activate

$env:FORCE_CLIP="1"

python build_index.py
```

CMD:

```cmd
cd /d D:\image_search
.\.venv\Scripts\activate

set FORCE_CLIP=1

python build_index.py
```

This generates:

```text
index/image_features.npy
index/image_paths.json
index/metadata.json
```

### Build with fallback encoder

PowerShell:

```powershell
$env:FORCE_CLIP="0"
python build_index.py
```

CMD:

```cmd
set FORCE_CLIP=
python build_index.py
```

## Important: Rebuild Index After Changing Encoder

If the previous index was built using the fallback encoder, but you now want to use real CLIP, you must delete the old index files and rebuild them.

PowerShell:

```powershell
Remove-Item -Force index\image_features.npy
Remove-Item -Force index\image_paths.json
Remove-Item -Force index\metadata.json

$env:FORCE_CLIP="1"
python build_index.py
```

CMD:

```cmd
del index\image_features.npy
del index\image_paths.json
del index\metadata.json

set FORCE_CLIP=1
python build_index.py
```

This step is necessary because fallback vectors and real CLIP vectors are not compatible.

## Run the Application

### Run with real CLIP

PowerShell:

```powershell
cd D:\image_search
.\.venv\Scripts\activate

$env:NO_PROXY="localhost,127.0.0.1,::1"
$env:no_proxy="localhost,127.0.0.1,::1"
$env:HTTP_PROXY=""
$env:HTTPS_PROXY=""
$env:ALL_PROXY=""

$env:FORCE_CLIP="1"

python app.py
```

CMD:

```cmd
cd /d D:\image_search
.\.venv\Scripts\activate

set NO_PROXY=localhost,127.0.0.1,::1
set no_proxy=localhost,127.0.0.1,::1
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=

set FORCE_CLIP=1

python app.py
```

Then open:

```text
http://127.0.0.1:7861
```

### Run with fallback encoder

PowerShell:

```powershell
$env:FORCE_CLIP="0"
python app.py
```

CMD:

```cmd
set FORCE_CLIP=
python app.py
```

## Interface Design

The interface follows the Five-Stage Search Framework.

| Stage | Implementation |
| --- | --- |
| Formulation | Users input text or upload an image |
| Preview | Users can preview their query |
| Initiation | Users click the Search button |
| Review | The system displays ranked image results and result overview |
| Refinement | Users can change Top-K and search again |
| Use | Users can download retrieved images as a ZIP file |

## How to Test

### Text-to-Image Search

Try queries such as:

```text
apple
banana
milk
bread
red fruit
bottle
```

### Image-to-Image Search

Upload an image from the dataset or a visually similar image, then click Search.

## Successful Real CLIP Startup

When the real CLIP model is used successfully, the terminal should show:

```text
[INFO] Loading real CLIP from local path: models\clip-vit-base-patch32
[INFO] Real CLIP model loaded successfully.
[INFO] Encoder status: Real CLIP
```

The web interface should also show:

```text
Current Encoder: Real CLIP
```

If the interface shows:

```text
Current Encoder: Fallback Encoder
```

then the real CLIP model is not being used.
