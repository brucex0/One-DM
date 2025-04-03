#!/usr/bin/env python3

import sys
import subprocess
import pkg_resources
import platform
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version {sys.version_info.major}.{sys.version_info.minor} detected")

def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        "data/IAM64-new",
        "data/IAM64_laplace",
        "model_zoo",
        "outputs/test"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating directory: {dir_path}")
            path.mkdir(parents=True, exist_ok=True)

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        "model_zoo/One-DM-ckpt.pt",
        "model_zoo/vae_HTR138.pth"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Warning: The following model files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please ensure these files are downloaded using the setup.sh script")

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow",
        "numpy",
        "opencv-python",
        "tensorboardX",
        "tqdm",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "einops",
        "easydict",
        "pyyaml",
        "omegaconf",
        "lmdb",
        "six",
        "packaging",
        "antlr4-python3-runtime",
        "gdown",  # For downloading from Google Drive
    ]

    print("Installing required packages...")
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
            print(f"{requirement} is already installed")
        except pkg_resources.DistributionNotFound:
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        except pkg_resources.VersionConflict:
            print(f"Upgrading {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", requirement])

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available. Detected {torch.cuda.device_count()} GPU(s)")
            print(f"CUDA version: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        else:
            print("Warning: CUDA is not available. Training will be very slow on CPU.")
    except ImportError:
        print("Warning: PyTorch is not installed yet. Will install it next.")

def setup_distributed_training():
    """Install additional packages for distributed training"""
    try:
        import torch.distributed
        print("Distributed training packages are available")
    except ImportError:
        print("Installing packages for distributed training...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.0.0"])

def main():
    print("Starting One-DM setup...")
    
    # Check Python version
    check_python_version()
    
    # Check directory structure
    check_directory_structure()
    
    # Check model files
    check_model_files()
    
    # Check CUDA availability
    check_cuda()
    
    # Install requirements
    install_requirements()
    
    # Setup distributed training
    setup_distributed_training()
    
    print("\nSetup completed successfully!")
    print("\nMake sure to:")
    print("1. Place your handwriting samples in data/IAM64-new/")
    print("2. Run the preprocessing script on your samples")
    print("3. Verify model files are in model_zoo/")
    print("\nTo start fine-tuning, use:")
    print("torchrun --nproc_per_node=2 train_finetune.py \\ ")
    print("    --stable_dif_path 'runwayml/stable-diffusion-v1-5' \\ ")
    print("    --cfg configs/IAM64_finetune.yml \\ ")
    print("    --one_dm 'model_zoo/One-DM-ckpt.pt' \\ ")
    print("    --ocr_model 'model_zoo/vae_HTR138.pth' \\ ")
    print("    --log 'finetune' \\ ")
    print("    --device cuda")

if __name__ == "__main__":
    main() 