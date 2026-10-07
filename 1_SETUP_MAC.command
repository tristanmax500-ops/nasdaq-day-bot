#!/bin/bash
# Nasdaq Bot - one-time setup (Mac / Linux)
cd "$(dirname "$0")"
echo ""
echo "=============================================="
echo "  Nasdaq Bot - one-time setup"
echo "=============================================="
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is not installed."
  echo "Download Python 3.12 from https://www.python.org/downloads/ , install it,"
  echo "then double-click this file again."
  read -p "Press Enter to close..."
  exit 1
fi
python3 -m venv venv || { echo "Could not create the environment."; read -p "Press Enter..."; exit 1; }
source venv/bin/activate
python -m pip install --upgrade pip
echo "Installing required packages (a minute or two)..."
pip install -r requirements.txt || { echo "Install failed - check internet and retry."; read -p "Press Enter..."; exit 1; }
echo "Installing the speed-up package (optional, ok if this fails)..."
pip install numba || true
chmod +x 2_START_MAC.command
echo ""
echo "=============================================="
echo "  Setup finished!  Now double-click 2_START_MAC.command"
echo "=============================================="
read -p "Press Enter to close..."
