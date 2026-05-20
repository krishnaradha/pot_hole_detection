# Pothole Detection — YOLOv8 on Azure ML

Train a YOLOv8 object-detection model to identify potholes in road images, using Azure Blob Storage for data and Azure ML for remote training.

## Project Structure

```
pot_hole_detection/
├── train.py              # YOLO training script (entry point for Azure ML job)
├── submit_job.py         # Submits the training job to Azure ML
├── environment.yml       # Conda environment definition
├── .env.example          # Template for required environment variables
├── configs/
│   └── data.yaml         # YOLO dataset config (paths resolved at runtime)
├── scripts/
│   ├── download_data.py  # Downloads dataset from Azure Blob Storage
│   └── upload_blob.py    # Uploads local images/labels to Azure Blob Storage
├── docker/
│   └── Dockerfile        # Azure ML container image definition
└── data/                 # Local dataset (gitignored — use scripts/download_data.py)
    ├── images/train/
    └── labels/train/
```

## Setup

### 1. Clone and configure environment

```bash
conda env create -f environment.yml
conda activate pothole_detection_env
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your Azure credentials
```

### 3. Download the dataset locally

```bash
python download_data.py
```

This pulls images and labels from Azure Blob Storage into `data/`.

## Training

### Local training

```bash
python train_local.py
```

Trains YOLOv8n for 50 epochs. Output (weights, metrics) is saved to `outputs/`.

### Remote training on Azure ML

```bash
python submit_pipeline.py
```

Submits a job to your Azure ML workspace. The training script runs on the configured compute cluster with data mounted from the registered datastore.

## Azure ML Prerequisites

- An Azure ML workspace with a compute cluster
- A registered datastore pointing to the blob container with your dataset
- The `COMPUTE_NAME` and `DATASTORE_NAME` set in `.env`

## Outputs

| File | Description |
|------|-------------|
| `outputs/best.pt` | Best model weights from training |
| `outputs/yolo_training/` | Training metrics, confusion matrix, batch previews |

## Dataset

- 40 labelled road images (`pothole_road_001.jpg` … `pothole_road_040.jpg`)
- Single class: `pothole`
- YOLO format annotations (`.txt` files with normalised bounding boxes)
