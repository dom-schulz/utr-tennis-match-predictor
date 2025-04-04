#!/bin/bash
set -e

# Install Docker
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Google Cloud SDK
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update
sudo apt-get install -y google-cloud-sdk

# Configure Docker to use Google Cloud Registry
sudo gcloud auth configure-docker us-west1-docker.pkg.dev

# Get credentials from Secret Manager
UTR_EMAIL=$(gcloud secrets versions access latest --secret="utr-email")
UTR_PASSWORD=$(gcloud secrets versions access latest --secret="utr-password")

# Pull the latest Docker image
sudo docker pull us-west1-docker.pkg.dev/cpsc324-project-452600/utr-scraper-repo/utr-scraper-image:latest

# Stop and remove any existing containers
sudo docker stop $(sudo docker ps -a -q --filter "name=utr-scraper" 2>/dev/null) 2>/dev/null || true
sudo docker rm $(sudo docker ps -a -q --filter "name=utr-scraper" 2>/dev/null) 2>/dev/null || true

# Starting container with image
echo "Starting container with image: us-west1-docker.pkg.dev/cpsc324-project-452600/utr-scraper-repo/utr-scraper-image:latest"
sudo docker run -d --name utr-scraper \
    -e UTR_EMAIL="$UTR_EMAIL" \
    -e UTR_PASSWORD="$UTR_PASSWORD" \
    -e GOOGLE_CLOUD_PROJECT="cpsc324-project-452600" \
    -e GCS_BUCKET_NAME="utr_scraper_bucket" \
    -e GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json" \
    us-west1-docker.pkg.dev/cpsc324-project-452600/utr-scraper-repo/utr-scraper-image:latest

# Output container logs
echo "Container started. To view logs, run: sudo docker logs utr-scraper" 