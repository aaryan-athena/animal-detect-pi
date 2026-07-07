#!/usr/bin/env bash
# Installs animal-detection as a systemd service so it starts automatically on Pi boot.
# Run this once on the Raspberry Pi: sudo bash deploy/install_service.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Please run as root: sudo bash deploy/install_service.sh" >&2
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-pi}"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Error: virtualenv python not found at $VENV_PYTHON" >&2
    echo "Create it first: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

SERVICE_FILE="/etc/systemd/system/animal-detection.service"

sed \
    -e "s#%USER%#$SERVICE_USER#g" \
    -e "s#%WORKDIR%#$PROJECT_DIR#g" \
    -e "s#%VENV_PYTHON%#$VENV_PYTHON#g" \
    "$PROJECT_DIR/deploy/animal-detection.service" > "$SERVICE_FILE"

systemctl daemon-reload
systemctl enable animal-detection
systemctl restart animal-detection

echo "Installed and started animal-detection service."
echo "Check status with: sudo systemctl status animal-detection"
echo "View logs with:    sudo journalctl -u animal-detection -f"
