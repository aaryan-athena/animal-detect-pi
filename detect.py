from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[WARN] RPi.GPIO not available. Buzzer functionality will be disabled.")


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
        "--buzzer-pin",
        type=int,
        default=17,
        help="GPIO pin number (BCM mode) for the piezo buzzer (default: 17).",
    )
    parser.add_argument(
        "--buzzer-duration",
        type=float,
        default=0.5,
        help="Duration in seconds for buzzer alert (default: 0.5).",
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
    parser.add_argument(
        "--no-save-images",
        dest="save_images",
        action="store_false",
        help="Disable saving a snapshot to disk on each detection.",
    )
    parser.set_defaults(save_images=True)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("captures"),
        help="Directory to store detection snapshots (default: captures/).",
    )
    parser.add_argument(
        "--save-cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between saved snapshots (default: 5.0).",
    )
    parser.add_argument(
        "--save-max-width",
        type=int,
        default=640,
        help="Resize snapshots so width does not exceed this many pixels, to keep files small (default: 640).",
    )
    parser.add_argument(
        "--save-quality",
        type=int,
        default=70,
        help="JPEG quality 1-100 for saved snapshots; lower = smaller files (default: 70).",
    )
    parser.add_argument(
        "--max-storage-mb",
        type=float,
        default=500.0,
        help="Prune oldest snapshots once --save-dir exceeds this size in MB. Set 0 to disable pruning (default: 500).",
    )
    return parser.parse_args()


def activate_buzzer(buzzer_pin: int, duration: float) -> None:
    """Activate the piezo buzzer on the specified GPIO pin."""
    if not GPIO_AVAILABLE:
        print("[ALERT] Animal detected! (Buzzer not available)")
        return
    
    try:
        GPIO.output(buzzer_pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(buzzer_pin, GPIO.LOW)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to activate buzzer: {exc}")


def play_sound(audio_path: Path | None, buzzer_pin: int | None = None, buzzer_duration: float = 0.5) -> None:
    # If on Raspberry Pi with GPIO, use the buzzer
    if GPIO_AVAILABLE and buzzer_pin is not None:
        activate_buzzer(buzzer_pin, buzzer_duration)
        return
    
    # Otherwise, try audio playback (for testing on other platforms)
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


def save_detection_snapshot(
    frame,
    save_dir: Path,
    detection_info: list[str],
    max_width: int,
    quality: int,
) -> Path | None:
    """Resize and JPEG-encode a detection frame to disk, keeping files small for Pi storage."""
    height, width = frame.shape[:2]
    if max_width > 0 and width > max_width:
        scale = max_width / width
        frame = cv2.resize(frame, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)

    now = datetime.now()
    day_dir = save_dir / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    classes = "-".join(info.split(" ")[0] for info in detection_info) or "animal"
    filename = f"{now.strftime('%H%M%S_%f')}_{classes}.jpg"
    out_path = day_dir / filename

    try:
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to save snapshot: {exc}")
        return None
    return out_path


def enforce_storage_limit(save_dir: Path, max_storage_mb: float) -> None:
    """Delete the oldest snapshots until save_dir is back under the configured size limit."""
    if max_storage_mb <= 0 or not save_dir.exists():
        return

    files = sorted(
        (p for p in save_dir.glob("**/*.jpg") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    total_bytes = sum(p.stat().st_size for p in files)
    limit_bytes = max_storage_mb * 1024 * 1024

    while total_bytes > limit_bytes and files:
        oldest = files.pop(0)
        try:
            size = oldest.stat().st_size
            oldest.unlink()
            total_bytes -= size
        except OSError as exc:
            print(f"[WARN] Failed to prune {oldest}: {exc}")


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

    # Initialize GPIO for buzzer if available
    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(args.buzzer_pin, GPIO.OUT)
        GPIO.output(args.buzzer_pin, GPIO.LOW)
        print(f"[INFO] GPIO initialized. Buzzer on pin {args.buzzer_pin}")

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
    last_save = 0.0

    if args.save_images:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Saving detection snapshots to {args.save_dir} (max width {args.save_max_width}px, quality {args.save_quality})")

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
            play_sound(args.audio, args.buzzer_pin, args.buzzer_duration)
            last_alert = now

        if detected and args.save_images and now - last_save >= args.save_cooldown:
            saved_path = save_detection_snapshot(
                annotated_frame,
                args.save_dir,
                detection_info,
                args.save_max_width,
                args.save_quality,
            )
            if saved_path is not None:
                print(f"[SAVED] {saved_path}")
                enforce_storage_limit(args.save_dir, args.max_storage_mb)
            last_save = now

        cv2.imshow("Animal Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Cleanup GPIO
    if GPIO_AVAILABLE:
        GPIO.cleanup()
        print("[INFO] GPIO cleaned up")


if __name__ == "__main__":
    main()
