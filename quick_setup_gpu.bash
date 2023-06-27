#!/bin/bash

working_dir = $( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$working_dir"

# Prompt for sudo password
read -sp "Enter your sudo password: " sudo_password
echo ""

# Update package lists and install python3-venv
echo "Install python"
echo $sudo_password | sudo -S apt update
echo $sudo_password | sudo -S apt install python3.11-venv
echo $sudo_password | sudo -S apt install python3-pip
echo $sudo_password | sudo -S apt install python-is-python3

echo "Installing venv"
pip install virtualenv
python3 -m venv .env

echo "Starting python environment"
source .env/bin/activate

echo "Installing pytorch"
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

echo "Installing package according to requirement"
cat requirements-gpu.txt | xargs -n 1 pip install

mkdir data
mkdir data/processed_norm