#!/bin/bash

# Check if Python is installed
if ! command -v python3 2>&1 /dev/null; then
    echo "Python is not installed. Installing Python..."
    sudo apt update
    sudo apt install python3 -y
fi

# Check if pip is installed
if ! command -v pip3 2>&1 /dev/null; then
    echo "Pip is not installed. Installing pip..."
    sudo apt update
    sudo apt install python3-pip -y
fi

# Check if git is installed
if ! command -v git 2>&1 /dev/null; then
    echo "Git is not installed. Installing git..."
    sudo apt update
    sudo apt install git -y
fi

#Check if mkdocs is installed
if ! command -v mkdocs 2>&1 /dev/null; then
    echo "Mkdocs is not installed. Installing git..."
    sudo apt update
    sudo apt install mkdocs -y
fi



# Create a Python virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate the virtual environment
venv_dir="venv"
if [ -f "${venv_dir}"/bin/activate ]
    then
        source "${venv_dir}"/bin/activate
        echo "Entering Virtual environment"
    else
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi

#Install Pytorch
echo "Installing Pytorch"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install required packages
if [ -f "requirements.txt" ]; then
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Clone ModelZoo
if [ ! -d "model_zoo" ]; then
    git clone https://HamnaAkram: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@huggingface.co/formsKorea/ModelZoo.git model_zoo
    echo "Copied Model zoo"

    else
    echo "Model Zoo already exists"
fi

#Install deepface

cd internal/components
if [ ! -d "deepface" ]; then
    git clone https://github.com/serengil/deepface.git
    cd deepface
    sudo pip install -e .
    echo "Installed deepface"

    else
    echo "Deepface already installed"
fi

cd ../../..

echo "the PWD is : ${PWD}"
mkdocs serve
echo "Setup completed successfully."

