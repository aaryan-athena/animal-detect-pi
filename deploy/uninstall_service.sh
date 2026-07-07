#!/usr/bin/env bash
# Removes the animal-detection systemd service.
# Run: sudo bash deploy/uninstall_service.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Please run as root: sudo bash deploy/uninstall_service.sh" >&2
    exit 1
fi

systemctl stop animal-detection || true
systemctl disable animal-detection || true
rm -f /etc/systemd/system/animal-detection.service
systemctl daemon-reload

echo "Removed animal-detection service."
