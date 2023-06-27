#!/bin/bash

# Change directory to the location of the script
cd "$(dirname "$0")"

# Prompt for sudo password
read -sp "Enter your sudo password: " sudo_password
echo ""

# Check Python version
python_version=$(python -V 2>&1 | awk '{print $2}')
if [[ $python_version =~ ^2\. ]]; then
    echo "Python 2 is not supported. Please install Python 3."
    exit 1
fi

# Update package lists and install python3-venv
echo "Install python"
echo $sudo_password | sudo apt update
echo $sudo_password | sudo apt install -y python3-venv
echo $sudo_password | sudo apt install -y python3-pip
echo $sudo_password | sudo apt install -y python-is-python3

echo "Installing venv"
python3 -m venv .env

echo "Starting python environment"
source .env/bin/activate

echo "Installing pytorch"
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

echo "Installing package according to requirement"
cat requirements-gpu.txt | xargs -n 1 pip install

mkdir data
mkdir data/processed_norm