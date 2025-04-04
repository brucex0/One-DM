#!/bin/bash

# Exit on error
set -e

echo "Starting Docker setup for One-DM..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo privileges"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
    echo "Detected OS: $OS $VERSION"
else
    echo "Cannot detect OS, assuming Debian/Ubuntu compatible"
    OS="debian"
fi

# Install Docker dependencies
echo "Installing Docker dependencies..."
apt-get update
apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    git \
    wget

# Add Docker's official GPG key
echo "Adding Docker GPG key..."
if [ ! -d "/etc/apt/keyrings" ]; then
    mkdir -p /etc/apt/keyrings
fi
curl -fsSL https://download.docker.com/linux/$OS/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the stable repository
echo "Setting up Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
echo "Installing Docker Engine..."
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io

# Add current user to docker group to avoid using sudo
echo "Adding $(logname) to the docker group..."
usermod -aG docker $(logname)

# Install NVIDIA CUDA and Container Toolkit
echo "Installing NVIDIA drivers and CUDA..."

# For Debian systems, we need to add non-free repositories
if [ "$OS" = "debian" ]; then
    echo "Adding non-free repositories for Debian..."
    apt-get install -y software-properties-common
    
    # Add non-free components to sources.list if they're not there already
    if ! grep -q "non-free" /etc/apt/sources.list; then
        sed -i 's/main/main contrib non-free non-free-firmware/g' /etc/apt/sources.list
    fi
    
    apt-get update
    
    # Install NVIDIA drivers
    apt-get install -y nvidia-driver firmware-misc-nonfree
    
    # Install CUDA via the NVIDIA repo
    echo "Installing CUDA from NVIDIA repository..."
    wget https://developer.download.nvidia.com/compute/cuda/12.3.1/local_installers/cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb
    dpkg -i cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb
    cp /var/cuda-repo-debian12-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda
    rm cuda-repo-debian12-12-3-local_12.3.1-545.23.08-1_amd64.deb
else
    # For Ubuntu, use the standard nvidia-cuda-toolkit
    apt-get install -y nvidia-cuda-toolkit
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
echo "Restarting Docker daemon..."
systemctl restart docker

# Create a project directory
PROJECT_DIR="/opt/one-dm"
echo "Creating project directory at ${PROJECT_DIR}..."
mkdir -p ${PROJECT_DIR}

# Clone the repository if not already cloned
if [ ! -d "${PROJECT_DIR}/One-DM" ]; then
    echo "Cloning One-DM repository..."
    git clone https://github.com/OPPO-Mente-Lab/One-DM.git ${PROJECT_DIR}/One-DM
else
    echo "One-DM repository already exists."
fi

# Set correct permissions
chown -R $(logname):$(logname) ${PROJECT_DIR}

# Go to the project directory
cd ${PROJECT_DIR}/One-DM

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/IAM64-new
mkdir -p data/IAM64_laplace
mkdir -p model_zoo
mkdir -p outputs/test

# Create Dockerfile if it doesn't exist
if [ ! -f "Dockerfile" ]; then
    echo "Creating Dockerfile..."
    cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app
COPY . /app

# Install git and other dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install additional Python packages
RUN pip install --no-cache-dir \
    gdown \
    opencv-python \
    tensorboard

# Create necessary directories
RUN mkdir -p data/IAM64-new && \
    mkdir -p data/IAM64_laplace && \
    mkdir -p model_zoo && \
    mkdir -p outputs/test

# Set up the entrypoint
ENTRYPOINT ["python", "train_finetune.py"]
EOF
fi

# Create build script
echo "Creating Docker build script..."
cat > build_docker.sh << 'EOF'
#!/bin/bash
set -e

# Configuration
PROJECT_ID="optimum-rock-447312-u8"  # Replace with your GCP project ID
REGION="us-west1"                    # Replace with your preferred region
IMAGE_NAME="one-dm-training"
IMAGE_TAG="latest"

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag for Google Container Registry if needed
if [[ "$1" == "--push" ]]; then
    echo "Tagging for GCR..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}
    
    # Configure Docker to use gcloud as a credential helper
    echo "Configuring Docker to use gcloud as a credential helper..."
    gcloud auth configure-docker
    
    # Push to GCR
    echo "Pushing to GCR..."
    docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}
    
    echo "Image pushed to: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo "Docker build complete!"
echo ""
echo "To run the image locally:"
echo "docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs ${IMAGE_NAME}:${IMAGE_TAG} [additional_args]"
EOF

chmod +x build_docker.sh

# Create test run script
echo "Creating Docker run script..."
cat > run_docker.sh << 'EOF'
#!/bin/bash
set -e

IMAGE_NAME="one-dm-training"
IMAGE_TAG="latest"

# Run the Docker container with GPU support
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/model_zoo:/app/model_zoo \
  ${IMAGE_NAME}:${IMAGE_TAG} \
  --stable_dif_path 'runwayml/stable-diffusion-v1-5' \
  --cfg configs/IAM64_finetune.yml \
  --one_dm 'model_zoo/One-DM-ckpt.pt' \
  --ocr_model 'model_zoo/vae_HTR138.pth' \
  --log 'finetune' \
  --device cuda
EOF

chmod +x run_docker.sh

echo "Setting correct permissions..."
chown $(logname):$(logname) Dockerfile build_docker.sh run_docker.sh

echo "Docker setup completed!"
echo ""
echo "IMPORTANT: You need to log out and log back in for the docker group membership to take effect."
echo ""
echo "Next steps:"
echo "1. Log out and log back in to apply docker group membership"
echo "2. Place your handwriting samples in ${PROJECT_DIR}/One-DM/data/IAM64-new"
echo "3. Download pre-trained models to ${PROJECT_DIR}/One-DM/model_zoo"
echo "4. Build the Docker image: cd ${PROJECT_DIR}/One-DM && ./build_docker.sh"
echo "5. Run training: ./run_docker.sh"
echo ""
echo "To push to Google Container Registry: ./build_docker.sh --push" 