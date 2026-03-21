#!/bin/bash

# VidCopilot Setup Script

echo "Setting up VidCopilot..."

# Install system dependencies
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y ffmpeg python3 python3-pip
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg python3
elif [[ "$OSTYPE" == "msys" ]]; then
    # Windows
    choco install ffmpeg python
fi

# Install Python dependencies
pip install -r requirements.txt

# Download models (example)
# wget -P models https://example.com/model.zip
# unzip models/model.zip

# Configure environment
echo "Setup complete. Run 'python agent/main.py --help' to get started."