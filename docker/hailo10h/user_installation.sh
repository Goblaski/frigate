#!/bin/bash

set -euxo pipefail

source /etc/os-release

if [[ "${VERSION_CODENAME}" != "trixie" ]]; then
    echo "This installer is intended for Raspberry Pi OS Trixie on Raspberry Pi 5 with AI HAT+ 2."
    echo "Detected VERSION_CODENAME=${VERSION_CODENAME}."
    exit 1
fi

sudo apt-get update
sudo apt-get install -y dkms hailo-h10-all

echo "Hailo-10H host installation complete. Reboot your system to load the new runtime and firmware."
echo "After reboot, verify with: hailortcli fw-control identify"
