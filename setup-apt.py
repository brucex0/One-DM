#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
import platform
import shutil
from pathlib import Path
import venv

class ExternallyManagedPythonSetup:
    def __init__(self, project_dir=None, cuda_version='11.8'):
        self.is_root = os.geteuid() == 0
        self.project_dir = project_dir or '/opt/one-dm'
        self.cuda_version = cuda_version
        self.env_dir = os.path.join(self.project_dir, 'One-DM', 'venv')
        self.username = os.environ.get('SUDO_USER', os.environ.get('USER'))
        
        # Colors for terminal output
        self.GREEN = '\033[92m'
        self.YELLOW = '\033[93m'
        self.RED = '\033[91m'
        self.RESET = '\033[0m'
        
    def print_color(self, color, message):
        print(f"{color}{message}{self.RESET}")
        
    def run_command(self, command, check=True, shell=False, sudo=False):
        """Run a command and return its output"""
        self.print_color(self.GREEN, f"Running: {' '.join(command) if isinstance(command, list) else command}")
        
        if sudo and not self.is_root:
            if isinstance(command, list):
                command = ['sudo'] + command
            else:
                command = f"sudo {command}"
                
        try:
            if shell or isinstance(command, str):
                result = subprocess.run(command, shell=True, check=check, text=True, 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(command, check=check, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result
        except subprocess.CalledProcessError as e:
            self.print_color(self.RED, f"Command failed: {e}")
            self.print_color(self.RED, f"Error output: {e.stderr}")
            if check:
                raise
            return e
    
    def check_requirements(self):
        """Check if system meets requirements"""
        self.print_color(self.GREEN, "Checking system requirements...")
        
        # Check if running on Debian
        if platform.system() != "Linux":
            self.print_color(self.RED, "This script is designed for Linux systems only.")
            return False
            
        try:
            os_info = self.run_command("cat /etc/os-release", shell=True)
            if "debian" not in os_info.stdout.lower():
                self.print_color(self.YELLOW, "Warning: This script is optimized for Debian-based systems.")
        except:
            self.print_color(self.YELLOW, "Warning: Unable to determine Linux distribution.")
            
        # Check Python version
        py_version = sys.version_info
        if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
            self.print_color(self.RED, f"Python 3.8+ is required. Found {py_version.major}.{py_version.minor}")
            return False
            
        # Check if we can create virtual environments
        if not shutil.which('python3') or not shutil.which('pip3'):
            self.print_color(self.RED, "Python3 and pip3 must be installed.")
            self.print_color(self.YELLOW, "Run: sudo apt-get install python3 python3-pip python3-venv")
            return False
            
        return True
    
    def fix_apt_sources(self):
        """Fix any broken apt sources before updating"""
        self.print_color(self.GREEN, "Checking for problematic apt sources...")
        
        # Check if Docker repo exists and is causing problems
        docker_source_path = "/etc/apt/sources.list.d/docker.list"
        if os.path.exists(docker_source_path):
            self.print_color(self.YELLOW, f"Found Docker repository sources that might be misconfigured")
            
            # Backup the file
            backup_cmd = f"sudo cp {docker_source_path} {docker_source_path}.bak"
            self.run_command(backup_cmd, shell=True, check=False)
            
            # Disable the problematic repositories by commenting them out
            disable_cmd = f"sudo sed -i 's/^deb/# deb/' {docker_source_path}"
            self.run_command(disable_cmd, shell=True, check=False)
            
            self.print_color(self.GREEN, f"Disabled potentially problematic Docker repository")
        
        # Check for other problematic repositories in the main sources list
        other_repos = [
            "download.docker.com"
        ]
        
        for repo in other_repos:
            check_cmd = f"grep -l '{repo}' /etc/apt/sources.list /etc/apt/sources.list.d/*.list 2>/dev/null || true"
            result = self.run_command(check_cmd, shell=True, check=False)
            
            if result.stdout.strip():
                files = result.stdout.strip().split('\n')
                for file_path in files:
                    if file_path and os.path.exists(file_path):
                        self.print_color(self.YELLOW, f"Disabling problematic repository in {file_path}")
                        backup_cmd = f"sudo cp {file_path} {file_path}.bak"
                        self.run_command(backup_cmd, shell=True, check=False)
                        
                        disable_cmd = f"sudo sed -i 's/^deb.*{repo}/# deb \\0/' {file_path}"
                        self.run_command(disable_cmd, shell=True, check=False)
        
    def install_system_packages(self):
        """Install required system packages"""
        if not self.is_root:
            self.print_color(self.YELLOW, "Non-root user detected. Using sudo for system package installation.")
        
        # Fix any problematic apt sources
        self.fix_apt_sources()
        
        packages = [
            "git",
            "build-essential",
            "wget",
            "curl",
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "ninja-build",
            "python3-venv",
            "python3-pip",
            "python3-dev"
        ]
        
        self.print_color(self.GREEN, "Updating package lists...")
        self.run_command(["apt-get", "update"], sudo=True)
        
        self.print_color(self.GREEN, f"Installing required packages: {' '.join(packages)}")
        self.run_command(["apt-get", "install", "-y"] + packages, sudo=True)
        
    def setup_project_directory(self):
        """Create project directory and clone repository"""
        # Create project directory if it doesn't exist
        self.print_color(self.GREEN, f"Setting up project directory at {self.project_dir}")
        
        if not os.path.exists(self.project_dir):
            self.run_command(["mkdir", "-p", self.project_dir], sudo=True)
            
        # Clone repository if not already cloned
        repo_dir = '.'#os.path.join(self.project_dir, "One-DM")
        if not os.path.exists(repo_dir):
            self.print_color(self.GREEN, "Cloning One-DM repository...")
            self.run_command(
                ["git", "clone", "https://github.com/brucex0/One-DM.git", repo_dir],
                sudo=True
            )
        else:
            self.print_color(self.YELLOW, "One-DM repository already exists. Skipping clone.")
            
        # Set correct permissions
        if self.username:
            self.print_color(self.GREEN, f"Setting ownership to {self.username}")
            self.run_command(["chown", "-R", f"{self.username}:{self.username}", self.project_dir], sudo=True)
            
        # Create necessary directories
        dirs = ["data/IAM64-new", "data/IAM64_laplace", "model_zoo", "outputs/test"]
        for d in dirs:
            dir_path = os.path.join(repo_dir, d)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                
    def create_virtual_env(self):
        """Create a virtual environment to bypass externally-managed-environment"""
        self.print_color(self.GREEN, f"Creating virtual environment at {self.env_dir}")
        
        if not os.path.exists(self.env_dir):
            venv.create(self.env_dir, with_pip=True)
            # Upgrade pip in the virtual environment
            pip_path = os.path.join(self.env_dir, "bin", "pip")
            self.run_command([pip_path, "install", "--upgrade", "pip", "setuptools", "wheel"])
        else:
            self.print_color(self.YELLOW, "Virtual environment already exists. Skipping creation.")
            
    def install_torch(self):
        """Install PyTorch with CUDA support"""
        pip_path = os.path.join(self.env_dir, "bin", "pip")
        
        self.print_color(self.GREEN, f"Installing PyTorch with CUDA {self.cuda_version} support...")
        if self.cuda_version == '11.8':
            self.run_command([
                pip_path, "install", "torch", "torchvision", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
        elif self.cuda_version == '11.6':
            self.run_command([
                pip_path, "install", "torch==1.13.1+cu116", "torchvision==0.14.1+cu116",
                "--index-url", "https://download.pytorch.org/whl/cu116"
            ])
        else:
            self.print_color(self.YELLOW, "Using default PyTorch version from PyPI (may not be optimized for your GPU)")
            self.run_command([pip_path, "install", "torch", "torchvision"])
            
    def install_dependencies(self):
        """Install required Python packages"""
        pip_path = os.path.join(self.env_dir, "bin", "pip")
        
        # Packages needed for preprocessing, downloading models, and training
        packages = [
            "gdown",           # For downloading pre-trained models
            "tensorboard",     # For visualizing training progress
            "opencv-python",   # For image processing
            "numpy",
            "pillow",
            "matplotlib",
            "tqdm"             # For progress bars
        ]
        
        self.print_color(self.GREEN, f"Installing Python dependencies: {', '.join(packages)}")
        self.run_command([pip_path, "install"] + packages)
            
    def create_activation_script(self):
        """Create a script to activate the environment"""
        activate_script = os.path.join(self.project_dir, "activate-one-dm.sh")
        
        self.print_color(self.GREEN, f"Creating activation script at {activate_script}")
        
        with open(activate_script, 'w') as f:
            f.write(f'''#!/bin/bash
# Activation script for One-DM environment
source {self.env_dir}/bin/activate
export PYTHONPATH={os.path.join(self.project_dir, "One-DM")}:$PYTHONPATH
echo "One-DM environment activated."
echo "For downloading models: python download_models.py"
echo "For preprocessing: python preprocess.py --input data/input --output data/IAM64-new"
echo "For training: python train_finetune.py --stable_dif_path 'runwayml/stable-diffusion-v1-5' ..."
''')
        
        os.chmod(activate_script, 0o755)
        
        # Create a symlink in /usr/local/bin if root
        if self.is_root:
            symlink_path = "/usr/local/bin/activate-one-dm"
            if os.path.exists(symlink_path):
                os.remove(symlink_path)
            os.symlink(activate_script, symlink_path)
        
    def create_download_script(self):
        """Create a Python script to download pre-trained models"""
        download_script = os.path.join(self.project_dir, "One-DM", "download_models.py")
        
        self.print_color(self.GREEN, f"Creating download models script at {download_script}")
        
        with open(download_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def download_models():
    """Download pre-trained models using gdown"""
    model_dir = Path("./model_zoo")
    model_dir.mkdir(exist_ok=True)
    
    # Check for gdown
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    print("Downloading pre-trained models...")
    
    # Google Drive folder with the models
    folder_url = "https://drive.google.com/drive/folders/1-0JQEfyNPuGKM1kCE-FUwgynhB0r-wP_"
    
    # Download the folder contents
    gdown.download_folder(folder_url, output=str(model_dir), quiet=False)
    
    print("Models downloaded to", model_dir)
    print("Available models:")
    for model_file in model_dir.glob("*"):
        print(f"- {model_file.name}")

if __name__ == "__main__":
    download_models()
''')
        
        os.chmod(download_script, 0o755)
    
    def create_preprocess_script(self):
        """Create preprocess script in the One-DM directory"""
        preprocess_script = os.path.join(self.project_dir, "One-DM", "preprocess.py")
        
        self.print_color(self.GREEN, f"Creating preprocessing script at {preprocess_script}")
        
        with open(preprocess_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

def preprocess_handwriting(input_path, output_path, target_size=64):
    """
    Preprocess handwriting image:
    1. Load the image
    2. Filter out white background
    3. Make strokes white on black background
    4. Resize while maintaining aspect ratio
    5. Center on black background
    
    Args:
        input_path: Path to input directory (can contain subfolders)
        output_path: Path to output directory (will mirror input structure)
        target_size: Target size for output images
    """
    # Convert paths to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Get all image files recursively
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg']:
        image_files.extend(list(input_path.rglob(f'*{ext}')))
        image_files.extend(list(input_path.rglob(f'*{ext.upper()}')))
    
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        # Calculate relative path from input directory
        rel_path = img_path.relative_to(input_path)
        
        # Create corresponding output directory
        output_dir = output_path / rel_path.parent
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}")
            continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to separate strokes from background
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Invert to get white strokes on black background
        inverted = cv2.bitwise_not(binary)
        
        # Find bounding box of content
        coords = cv2.findNonZero(inverted)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            inverted = inverted[y:y+h, x:x+w]
        
        # Convert to PIL Image for better resizing
        pil_img = Image.fromarray(inverted)
        
        # Calculate new size while maintaining aspect ratio
        ratio = target_size / max(pil_img.size)
        new_size = tuple([int(x * ratio) for x in pil_img.size])
        resized = pil_img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new black image
        final = Image.new('L', (target_size, target_size), 0)
        
        # Paste resized image in center
        paste_x = (target_size - new_size[0]) // 2
        paste_y = (target_size - new_size[1]) // 2
        final.paste(resized, (paste_x, paste_y))
        
        # Save preprocessed image maintaining folder structure
        output_file = output_path / rel_path
        final.save(output_file)
        print(f"Processed {rel_path}")

def setup_directories():
    """Create input and output directories if they don't exist"""
    input_dir = Path("data/input")
    output_dir = Path("data/IAM64-new")
    
    input_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    return input_dir, output_dir

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess handwriting samples')
    parser.add_argument('--input', type=str, default='data/input',
                        help='Input directory containing handwriting images')
    parser.add_argument('--output', type=str, default='data/IAM64-new',
                        help='Output directory for preprocessed images')
    parser.add_argument('--size', type=int, default=64,
                        help='Target size for output images (default: 64)')
    
    args = parser.parse_args()
    
    input_dir, output_dir = setup_directories()
    
    # Check if input directory has files
    image_count = 0
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        image_count += len(list(Path(args.input).rglob(f'*{ext}')))
    
    if image_count == 0:
        print(f"No images found in {args.input}")
        print(f"Please place your handwriting samples in the {args.input} directory.")
        print("Example structure:")
        print(f"  {args.input}/")
        print(f"  ├── user1/")
        print(f"  │   ├── sample1.png")
        print(f"  │   └── sample2.png")
        print(f"  └── user2/")
        print(f"      ├── sample1.png")
        print(f"      └── sample2.png")
        exit(1)
    
    preprocess_handwriting(args.input, args.output, args.size)
    print(f"Preprocessing complete! Images saved to {args.output}")
''')
        
        os.chmod(preprocess_script, 0o755)
    
    def run(self):
        """Run the complete setup process"""
        if not self.check_requirements():
            self.print_color(self.RED, "System requirements not met. Exiting.")
            return False
            
        try:
            self.print_color(self.GREEN, "Starting One-DM setup...")
            
            self.install_system_packages()
            self.setup_project_directory()
            self.create_virtual_env()
            self.install_torch()
            self.install_dependencies()
            self.create_activation_script()
            self.create_download_script()
            self.create_preprocess_script()
            
            self.print_color(self.GREEN, "\n✅ Setup completed successfully!\n")
            
            # Print usage instructions
            self.print_color(self.GREEN, "To use One-DM:")
            print(f"1. Activate the environment:")
            print(f"   source {os.path.join(self.project_dir, 'activate-one-dm.sh')}")
            print(f"2. Change to the One-DM directory:")
            print(f"   cd {os.path.join(self.project_dir, 'One-DM')}")
            print(f"3. Download the pre-trained models:")
            print(f"   python download_models.py")
            print(f"4. Place your handwriting samples in 'data/input'")
            print(f"5. Preprocess your samples:")
            print(f"   python preprocess.py")
            print(f"6. Start training:")
            print(f"   python train_finetune.py --stable_dif_path 'runwayml/stable-diffusion-v1-5' \\" )
            print(f"      --cfg configs/IAM64_finetune.yml \\" )
            print(f"      --one_dm 'model_zoo/One-DM-ckpt.pt' \\" )
            print(f"      --ocr_model 'model_zoo/vae_HTR138.pth' \\" )
            print(f"      --log 'finetune' \\" )
            print(f"      --device cuda" )
            
            return True
            
        except Exception as e:
            self.print_color(self.RED, f"Error during setup: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
def main():
    parser = argparse.ArgumentParser(description='Set up One-DM on a Debian system with externally managed Python')
    parser.add_argument('--dir', type=str, default='/opt/one-dm',
                        help='Installation directory (default: /opt/one-dm)')
    parser.add_argument('--cuda', type=str, default='11.8',
                        help='CUDA version for PyTorch (default: 11.8)')
    
    args = parser.parse_args()
    
    setup = ExternallyManagedPythonSetup(project_dir=args.dir, cuda_version=args.cuda)
    success = setup.run()
    
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    main() 