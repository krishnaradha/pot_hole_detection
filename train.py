import argparse
import os
import shutil
import yaml
from datetime import datetime

from ultralytics import YOLO

DEFAULT_DATA_CONFIG = "configs/data_local.yaml"


def find_dataset_base():
    """Locate the dynamic AzureML INPUT_data mount path (since run_id changes each run)."""
    azureml_base = "/mnt/azureml"
    for root, dirs, _ in os.walk(azureml_base):
        if "INPUT_data" in dirs:
            found_path = os.path.join(root, "INPUT_data")
            print(f"Found dataset base: {found_path}")
            return found_path
    raise FileNotFoundError("Could not find INPUT_data folder under /mnt/azureml.")


def create_azure_data_yaml(base_dir, output_path="outputs/data.yaml"):
    """Write a YOLO data.yaml resolved to the Azure ML mount path."""
    data = {
        "train": os.path.join(base_dir, "images", "train"),
        "val": os.path.join(base_dir, "images", "train"),
        "nc": 1,
        "names": ["pothole"],
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f)
    print(f"Created Azure data.yaml at {output_path}")
    return output_path


def resolve_data_config(data_arg):
    """Return the data config path to use.

    Uses the supplied path if the file exists.
    Falls back to Azure ML dynamic detection otherwise (remote job context).
    """
    if os.path.exists(data_arg):
        print(f"Using local data config: {data_arg}")
        return data_arg

    print(f"Config '{data_arg}' not found — detecting Azure ML mount path...")
    # Install OpenCV system dependencies required inside the Azure ML container
    os.system("apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0")
    dataset_base = find_dataset_base()
    return create_azure_data_yaml(dataset_base)


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 pothole detection training")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_CONFIG,
        help="Path to YOLO data config YAML (default: configs/data_local.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    print("Starting YOLO training...")
    original_dir = os.getcwd()

    try:
        data_yaml_path = resolve_data_config(args.data)

        run_name = f"yolo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        outputs_dir = os.path.join(original_dir, "outputs")
        best_src = os.path.join(outputs_dir, run_name, "weights", "best.pt")
        best_dest = os.path.join(outputs_dir, f"best_{run_name}.pt")

        model = YOLO("yolov8n.pt")
        model.train(
            data=data_yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=outputs_dir,
            name=run_name,
            exist_ok=False,
        )

        if os.path.exists(best_src):
            shutil.copy2(best_src, best_dest)
            print(f"Best model saved to {best_dest}")

    finally:
        os.chdir(original_dir)
        print(f"Returned to original directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
