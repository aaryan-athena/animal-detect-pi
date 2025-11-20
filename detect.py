from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live YOLO inference and play a sound when an animal is detected."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to the trained YOLO weights file. If omitted, the latest weights in runs/ will be used."
        ),
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: camera index (e.g. '0') or path to a video/RTSP stream.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for detections (default: 0.35).",
    )
    parser.add_argument(
        "--deer-conf",
        type=float,
        default=0.65,
        help="Optional higher confidence threshold specifically for deer class to reduce false positives.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Optional path to a WAV audio file to play when an animal is detected.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Minimum seconds between alert sounds (default: 2.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device identifier passed to Ultralytics (e.g. '0', 'cpu').",
    )
    return parser.parse_args()


def play_sound(audio_path: Path | None) -> None:
    if audio_path and audio_path.exists():
        try:
            if sys.platform.startswith("win"):
                import winsound

                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                from playsound import playsound  # type: ignore[import-not-found]

                playsound(str(audio_path), block=False)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to play custom audio: {exc}")

    # Fallback to a short beep if no audio file is provided or playback fails.
    if sys.platform.startswith("win"):
        import winsound

        winsound.Beep(1200, 180)
    else:
        print("[ALERT] Animal detected!")


def open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    print(f"[INFO] Camera found and opened successfully: {source}")
    return cap


def find_fallback_weights(preferred: Path | None = None) -> Path | None:
    # Try best.pt then last.pt in the same run directory hierarchy.
    runs_root = Path("runs")
    if not runs_root.exists():
        return None

    candidates: list[Path] = []
    for pattern in ("**/weights/best.pt", "**/weights/last.pt"):
        candidates.extend(runs_root.glob(pattern))

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for candidate in candidates:
        if preferred is None or candidate.resolve() != preferred.resolve():
            return candidate
    return None


def summarize_available_weights() -> str:
    runs_root = Path("runs")
    if not runs_root.exists():
        return "No runs directory found."
    entries = []
    for weight in runs_root.glob("**/weights/*.pt"):
        rel = weight.relative_to(runs_root.parent if runs_root.parent != Path(".") else Path())
        entries.append(f"- {rel}")
    if not entries:
        return "No weight files found under runs/."
    entries.sort()
    return "\n".join(entries)


def main() -> None:
    args = parse_args()

    model_path: Path | None = args.model
    if model_path is not None and not model_path.exists():
        fallback = find_fallback_weights(model_path)
        if fallback:
            print(f"[INFO] Requested weights not found; using latest available at {fallback}")
            model_path = fallback
        else:
            available = summarize_available_weights()
            raise FileNotFoundError(
                f"Model weights not found at {model_path}.\nAvailable weights:\n{available}"
            )

    if model_path is None:
        model_path = find_fallback_weights()
        if model_path:
            print(f"[INFO] Using latest available weights at {model_path}")
        else:
            available = summarize_available_weights()
            raise FileNotFoundError(
                "No weight file provided and none discovered under runs/.\n"
                f"Available weights:\n{available}"
            )

    model = YOLO(str(model_path))
    cap = open_capture(args.source)
    last_alert = 0.0

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] End of stream or camera disconnected.")
            break

        results = model.predict(
            frame,
            conf=args.conf,
            device=args.device,
            stream=False,
            verbose=False,
        )

        annotated_frame = frame
        detected = False
        detection_info = []
        for result in results:
            if result.boxes and len(result.boxes) > 0:
                # Filter boxes based on class-specific confidence thresholds
                valid_boxes = []
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = result.names[cls] if result.names else f"class_{cls}"
                    
                    # Apply higher threshold for deer class if specified
                    if args.deer_conf is not None and class_name == "deers" and conf < args.deer_conf:
                        continue
                    
                    valid_boxes.append(box)
                    detection_info.append(f"{class_name} ({conf:.2f})")
                
                if valid_boxes:
                    detected = True
                    annotated_frame = result.plot()

        if detection_info:
            print(f"[DETECTION] {', '.join(detection_info)}")

        now = time.time()
        if detected and now - last_alert >= args.cooldown:
            play_sound(args.audio)
            last_alert = now

        cv2.imshow("Animal Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
