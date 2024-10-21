#!/bin/bash

# Update package lists
sudo apt-get update

# Install libGL for OpenCV
sudo apt-get install -y libgl1-mesa-glx

# Install other system dependencies (if any)
# sudo apt-get install -y <other-dependencies>

# Install Python dependencies from requirements.txt
pip install -r requirements.txt
