#!/bin/bash
cd "$(dirname "$0")"
if [ ! -f venv/bin/activate ]; then
  echo "Please run 1_SETUP_MAC.command first."
  read -p "Press Enter to close..."
  exit 1
fi
source venv/bin/activate
python run.py "$@"
