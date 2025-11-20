Animal Detection V2

YOLO-based real-time wildlife detection system
Detects five animal species: buffaloes, deer, elephants, rhinos, tigers from images or live camera feeds.
Includes dataset preparation tools, YOLO training scripts, and live inference with audio alerts.

## Dataset Details

### Class Distribution
The dataset contains 358 total annotations across 5 animal classes:

| Class       | Count | Percentage |
|-------------|-------|------------|
| Rhinos      | 127   | 35.5%      |
| Deers       | 97    | 27.1%      |
| Tigers      | 68    | 19.0%      |
| Elephants   | 50    | 14.0%      |
| Buffaloes   | 16    | 4.5%       |

**Note:** The deer class may be prone to overfitting due to its high representation. Use `--deer-conf` parameter during inference to set a higher confidence threshold for deer detections.

### Training Configuration
- **Base Model:** YOLOv8n (nano) - optimized for edge devices
- **Image Size:** 640x640
- **Epochs:** 100 (recommended: 150 for better generalization)
- **Batch Size:** 16
- **Augmentation:** HSV, rotation, translation, scale, flip, mosaic, mixup
- **Split Ratio:** 80% train, 10% validation, 10% test

Setup
Create and Activate a Python Environment
python -m venv .venv
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

Dataset Preparation

Split the flat dataset into the Ultralytics YOLO directory format:

python prepare_dataset.py

Optional Arguments:

--ratios 0.7 0.2 0.1 → custom train/val/test split

--seed 123 → fixed shuffling

--move → move files instead of copying

Output directory:

datasets/animals/

 Training the YOLO Model

Fine-tune a YOLO checkpoint:

python train_yolo.py --model yolov8n.pt --device 0 --epochs 100

Useful Flags

--batch 16 → batch size

--imgsz 640 → training resolution

--project runs/animals

--name yolov8n-animals

--resume → continue previous training

Outputs

Trained weights are saved at:

runs/animals/<run-name>/weights/{best.pt, last.pt}

🎥 Real-Time Detection + Audio Alerts

Run inference from webcam/video and trigger an alert on detection:

python detect_and_alert.py --source 0 --audio path\to\alert.wav

Arguments

--model runs/animals/yolov8n-animals3/weights/best.pt

--conf 0.35 → confidence threshold

--deer-conf 0.65 → higher threshold for deer class (reduces false positives)

--cooldown 2.0 → minimum delay between alerts

--device 0 → GPU (cpu fallback available)

If no model path is provided, the script automatically selects the latest weights from:

runs/animals/**/weights/

### Console Logging
The detection script now includes console logs for headless operation:
- `[INFO] Camera found and opened successfully` - confirms camera initialization
- `[DETECTION] deers (0.87), tigers (0.92)` - logs each detection with confidence scores

 Dependencies

Installed via:

pip install -r requirements.txt


Includes:

ultralytics → YOLO training & inference

opencv-python → video I/O

playsound → audio notifications

polars → fast logging backend

Make sure NVIDIA drivers + CUDA runtime are installed if training on GPU.

## 🍓 Raspberry Pi Deployment

### Required Files for Production
To deploy on Raspberry Pi, you only need these files:

```
animal-detection-v2/
├── detect_and_alert.py          # Main inference script
├── requirements.txt             # Python dependencies
├── animals.yaml                 # Dataset config (for class names)
├── alarm/
│   └── audio.wav               # Alert sound file
└── runs/
    └── animals/
        └── <your-run>/
            └── weights/
                └── best.pt     # Trained model weights
```

### Installation on Raspberry Pi

1. **Install Python dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python-headless
```

2. **Enable camera:**
```bash
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

3. **Run detection (headless mode):**
```bash
python detect_and_alert.py --source 0 --conf 0.35 --deer-conf 0.65 --device cpu
```

4. **Monitor via SSH:**
All detections will be logged to console with timestamps and confidence scores.

### Performance Tips for Raspberry Pi
- Use `--device cpu` (no GPU on Pi)
- Consider YOLOv8n (nano) for best performance
- Lower resolution if needed: add custom imgsz parameter
- Use `opencv-python-headless` to reduce dependencies
- Disable GUI display for headless operation (already handled in script)

### Optional: Run as systemd service
Create `/etc/systemd/system/animal-detection.service`:
```ini
[Unit]
Description=Animal Detection Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/animal-detection-v2
ExecStart=/home/pi/animal-detection-v2/venv/bin/python detect_and_alert.py --source 0 --conf 0.35 --deer-conf 0.65 --device cpu
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable animal-detection
sudo systemctl start animal-detection
sudo journalctl -u animal-detection -f  # View logs
```

## Useful Tips

Check model performance, predictions, and logs in:

runs/animals/


Validate on test set:

yolo val


For offline video analysis:

python detect_and_alert.py --source path\to\video.mp4


Use --move carefully in dataset script (files will be permanently relocated).

### Reducing Deer False Positives
If deer detections are too frequent:
```bash
python detect_and_alert.py --deer-conf 0.70  # Increase threshold
```
