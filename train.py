import argparse
import os
import yaml
os.system("apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0")
from ultralytics import YOLO

# Ensure dependencies (for OpenCV image handling)

def find_dataset_base():
    """Locate the dynamic AzureML INPUT_data mount path (since run_id changes each run)."""
    azureml_base = "/mnt/azureml"
    for root, dirs, _ in os.walk(azureml_base):
        print(f"Checking {root} for INPUT_data...{dirs}")
        if "INPUT_data" in dirs:
            found_path = os.path.join(root, "INPUT_data")
            print(f"✅ Found dataset base: {found_path}")
            return found_path
    raise FileNotFoundError("❌ Could not find INPUT_data folder under /mnt/azureml.")


def create_data_yaml(base_dir, output_path="outputs/data.yaml"):
    """Create a YOLO data.yaml dynamically based on detected dataset path."""
    data = {
        "train": os.path.join(base_dir, "images", "train"),
        "val": os.path.join(base_dir, "images", "train"),
        "nc": 1,
        "names": ["pothole"]
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f)
    print(f"✅ Created data.yaml at {output_path}")
    print(f"Contents:\n{yaml.dump(data)}")
    return output_path


def main():
    print("Starting YOLO training...")

    # Save original working directory
    original_dir = os.getcwd()

    try:
        # Detect dataset base path dynamically
        dataset_base = find_dataset_base()

        # Create a fresh data.yaml dynamically
        data_yaml_path = create_data_yaml(dataset_base)

        # Load YOLOv8 model
        model = YOLO("yolov8n.pt")  # lightweight for quick training/testing

        # Start training
        results = model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=640,
            batch=8,
            project="outputs",
            name="pothole_detection_training",
            exist_ok=True
        )

        # Move the best weights for Azure ML artifact tracking
        best_model_path = os.path.join("outputs", "pothole_detection_training", "weights", "best.pt")
        if os.path.exists(best_model_path):
            os.rename(best_model_path, "outputs/best.pt")
            print("✅ Best model saved to outputs/best.pt")

    finally:
        # Always go back to the original directory
        os.chdir(original_dir)
        print(f"Returned to original directory: {os.getcwd()}")


if __name__ == "__main__":
    main()
