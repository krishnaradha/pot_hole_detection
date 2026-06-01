import argparse
import os
from pathlib import Path

from ultralytics import YOLO

DEFAULT_SOURCE = "data/images/train"
DEFAULT_OUTPUT = "outputs/inference"


def find_latest_model(search_dir: str = "outputs") -> str:
    """Return the most recently modified best_*.pt in search_dir (top-level copies)."""
    root = Path(search_dir)
    candidates = sorted(
        root.glob("best_*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No best_*.pt found in '{search_dir}/'")
    return str(candidates[0])


def run_inference(model_path: str | None, source: str, output_dir: str, conf: float) -> None:
    if model_path is None:
        model_path = find_latest_model()
        print(f"Auto-selected model: {model_path}")
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source not found: {source}")

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path)
    print(f"Model:   {model_path}")
    print(f"Source:  {source}")
    print(f"Output:  {output_path}")
    print(f"Conf:    {conf}\n")

    results = model.predict(source=source, conf=conf, verbose=False)

    print(f"{'─' * 52}")
    print(f"{'Image':<40} {'Detections':>12}")
    print(f"{'─' * 52}")

    total_detections = 0
    for r in results:
        name = Path(r.path).name
        count = len(r.boxes)
        total_detections += count
        status = f"{count} pothole(s)" if count else "none"
        print(f"{name:<40} {status:>12}")

        # Save annotated image directly to output_dir
        r.save(filename=str(output_path / name))

    print(f"{'─' * 52}")
    print(f"Total detections : {total_detections}")
    print(f"\nAnnotated images → {output_path}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 pothole inference on local images")
    parser.add_argument("--model", default=None, help="Path to model weights (.pt) — defaults to latest best.pt in outputs/")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Image file or folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0–1)")
    args = parser.parse_args()

    run_inference(args.model, args.source, args.output, args.conf)


if __name__ == "__main__":
    main()
