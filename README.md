# HCI Lab 3: Image Search System

This project implements an image search interface using Gradio and CLIP.

## Features

- Text-to-Image Search
- Image-to-Image Search
- Ranked results by cosine similarity
- Top-K result control
- Query preview
- Result overview
- Download retrieved images as a ZIP file

## Environment

Python 3.10 is recommended. The project can also run on newer Python versions when all dependencies are available.

## Installation

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If you use conda:

```bash
conda create -n hci_lab3 python=3.10 -y
conda activate hci_lab3
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

In this workspace, a local virtual environment has been created:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Prepare Dataset

Put image files into:

```text
data/images
```

Supported formats:

```text
jpg, jpeg, png, bmp, webp
```

## Build Image Index

```bash
python build_index.py
```

Or with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe build_index.py
```

This generates:

```text
index/image_features.npy
index/image_paths.json
index/metadata.json
```

## Run the Application

```bash
python app.py
```

Or with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Encoder Mode

By default, the app tries to use a locally cached CLIP model. If the CLIP model is not available locally, it falls back to a lightweight local encoder so the project can still run offline.

To force downloading and using the real CLIP model:

```powershell
$env:IMAGE_SEARCH_ENCODER="clip"
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe app.py
```

To explicitly use the offline fallback:

```powershell
$env:IMAGE_SEARCH_ENCODER="fallback"
.\.venv\Scripts\python.exe build_index.py
.\.venv\Scripts\python.exe app.py
```

## Project Structure

```text
hci_lab3_image_search/
├── app.py
├── build_index.py
├── clip_utils.py
├── search_utils.py
├── requirements.txt
├── README.md
├── data/images/
├── index/
└── outputs/
```
