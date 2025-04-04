#!/bin/bash

# Exit on error
set -e

# Configuration
PROJECT_ID="optimum-rock-447312-u8"  # Replace with your GCP project ID
REGION="us-west1"          # Replace with your preferred region
BUCKET_NAME="one-dm-training-data"  # Replace with your GCS bucket name
IMAGE_NAME="one-dm-training"
IMAGE_TAG="latest"
JOB_NAME="one-dm-training-$(date +%Y%m%d-%H%M%S)"

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

# Check if Vertex AI API is enabled
if ! gcloud services list --enabled | grep -q "aiplatform.googleapis.com"; then
    echo "Error: Vertex AI API is not enabled. Please enable it first."
    echo "Run: gcloud services enable aiplatform.googleapis.com"
    exit 1
fi

# Check if bucket exists, create if it doesn't
if ! gsutil ls -b gs://${BUCKET_NAME} &> /dev/null; then
    echo "Creating GCS bucket: gs://${BUCKET_NAME}"
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
fi

# Submit the training job
echo "Submitting training job: ${JOB_NAME}"
gcloud ai jobs submit training ${JOB_NAME} \
    --region=${REGION} \
    --master-image-uri=gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} \
    --job-dir=gs://${BUCKET_NAME}/${JOB_NAME} \
    --machine-type=n1-highmem-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --scale-tier=CUSTOM \
    --python-module=train \
    --args="--config=configs/train.yaml" \
    --time-limit=24h

echo "Training job submitted successfully: ${JOB_NAME}"
echo ""
echo "Next steps:"
echo "1. Use upload_samples.sh to upload your raw samples to the job"
echo "2. Monitor the training progress using:"
echo "   gcloud ai jobs describe ${JOB_NAME} --region=${REGION}"
echo "3. View logs using:"
echo "   gcloud ai jobs stream-logs ${JOB_NAME} --region=${REGION}" 