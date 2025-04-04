#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID="optimum-rock-447312-u8"  # Replace with your GCP project ID
REGION="us-west1"          # Replace with your preferred region
IMAGE_NAME="one-dm-training"
IMAGE_TAG="latest"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if gcloud is installed and configured
if ! command -v gcloud &> /dev/null; then
    echo "Error: Google Cloud SDK is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated with gcloud
if ! gcloud auth print-identity-token &> /dev/null; then
    echo "Error: Not authenticated with Google Cloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Configure Docker to use gcloud as a credential helper
echo "Configuring Docker to use gcloud as a credential helper..."
gcloud auth configure-docker

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Tag the image for GCR
echo "Tagging image for GCR..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}

# Push the image to GCR
echo "Pushing image to GCR..."
docker push gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}

echo "Image successfully pushed to GCR: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "Next steps:"
echo "1. Update submit_training.sh with your project ID and bucket name"
echo "2. Run submit_training.sh to start the training job"
echo "3. Use upload_samples.sh to upload your raw samples to the job" 