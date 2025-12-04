from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# GPIO backend selection: lgpio (best for Pi 5), then gpiozero, then RPi.GPIO
GPIO_AVAILABLE = False
GPIO_BACKEND = None
_lgpio_handle = None

# Try lgpio first (recommended for Raspberry Pi 5 on both OS 12 and OS 13)
try:
    import lgpio
    GPIO_AVAILABLE = True
    GPIO_BACKEND = "lgpio"
except ImportError:
    pass

# Fall back to gpiozero
if not GPIO_AVAILABLE:
    try:
        from gpiozero import Buzzer
        from gpiozero.exc import BadPinFactory
        GPIO_AVAILABLE = True
        GPIO_BACKEND = "gpiozero"
    except ImportError:
        pass

# Fall back to RPi.GPIO (won't work on Pi 5, but kept for older Pi models)
if not GPIO_AVAILABLE:
    try:
        import RPi.GPIO as GPIO
        GPIO_AVAILABLE = True
        GPIO_BACKEND = "rpigpio"
    except ImportError:
        pass

if not GPIO_AVAILABLE:
    print("[WARN] No GPIO library available (lgpio, gpiozero, or RPi.GPIO). Buzzer functionality will be disabled.")


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
        "--headless",
        action="store_true",
        help="Run without display window (for SSH/remote use).",
    )
    return parser.parse_args()


# Global buzzer instance for gpiozero
_buzzer_instance = None
_buzzer_pin_initialized = None


def init_buzzer(buzzer_pin: int) -> None:
    """Initialize the buzzer based on available GPIO backend."""
    global _buzzer_instance, _lgpio_handle, _buzzer_pin_initialized
    
    if not GPIO_AVAILABLE:
        return
    
    if GPIO_BACKEND == "lgpio":
        try:
            _lgpio_handle = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(_lgpio_handle, buzzer_pin)
            lgpio.gpio_write(_lgpio_handle, buzzer_pin, 0)
            _buzzer_pin_initialized = buzzer_pin
            print(f"[INFO] lgpio buzzer initialized on pin {buzzer_pin}")
        except Exception as exc:
            print(f"[WARN] Failed to initialize lgpio buzzer: {exc}")
    elif GPIO_BACKEND == "gpiozero":
        try:
            _buzzer_instance = Buzzer(buzzer_pin)
            print(f"[INFO] gpiozero buzzer initialized on pin {buzzer_pin}")
        except BadPinFactory as exc:
            print(f"[WARN] Failed to initialize gpiozero buzzer: {exc}")
    elif GPIO_BACKEND == "rpigpio":
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(buzzer_pin, GPIO.OUT)
        GPIO.output(buzzer_pin, GPIO.LOW)
        print(f"[INFO] RPi.GPIO buzzer initialized on pin {buzzer_pin}")


def cleanup_buzzer() -> None:
    """Cleanup GPIO resources."""
    global _buzzer_instance, _lgpio_handle, _buzzer_pin_initialized
    
    if GPIO_BACKEND == "lgpio" and _lgpio_handle is not None:
        try:
            if _buzzer_pin_initialized is not None:
                lgpio.gpio_write(_lgpio_handle, _buzzer_pin_initialized, 0)
            lgpio.gpiochip_close(_lgpio_handle)
            _lgpio_handle = None
            _buzzer_pin_initialized = None
            print("[INFO] lgpio buzzer cleaned up")
        except Exception as exc:
            print(f"[WARN] Failed to cleanup lgpio: {exc}")
    elif GPIO_BACKEND == "gpiozero" and _buzzer_instance:
        _buzzer_instance.close()
        _buzzer_instance = None
        print("[INFO] gpiozero buzzer cleaned up")
    elif GPIO_BACKEND == "rpigpio":
        GPIO.cleanup()
        print("[INFO] RPi.GPIO cleaned up")


def activate_buzzer(buzzer_pin: int, duration: float) -> None:
    """Activate the piezo buzzer on the specified GPIO pin."""
    global _buzzer_instance, _lgpio_handle
    
    if not GPIO_AVAILABLE:
        print("[ALERT] Animal detected! (Buzzer not available)")
        return
    
    try:
        if GPIO_BACKEND == "lgpio" and _lgpio_handle is not None:
            lgpio.gpio_write(_lgpio_handle, buzzer_pin, 1)
            time.sleep(duration)
            lgpio.gpio_write(_lgpio_handle, buzzer_pin, 0)
        elif GPIO_BACKEND == "gpiozero" and _buzzer_instance:
            _buzzer_instance.on()
            time.sleep(duration)
            _buzzer_instance.off()
        elif GPIO_BACKEND == "rpigpio":
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

    # Initialize buzzer if GPIO available
    if GPIO_AVAILABLE:
        init_buzzer(args.buzzer_pin)

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
            play_sound(args.audio, args.buzzer_pin, args.buzzer_duration)
            last_alert = now

        if not args.headless:
            cv2.imshow("Animal Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # In headless mode, just check for keyboard interrupt
            pass

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    
    # Cleanup GPIO
    if GPIO_AVAILABLE:
        cleanup_buzzer()


if __name__ == "__main__":
    main()
