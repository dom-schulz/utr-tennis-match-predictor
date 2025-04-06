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

# Remove any previous shutdown signal if it exists
gsutil rm -f gs://utr_scraper_bucket/shutdown_signal.txt 2>/dev/null || true
echo "Removed any previous shutdown signals"

# Starting container with image
echo "Starting container with image: us-west1-docker.pkg.dev/cpsc324-project-452600/utr-scraper-repo/utr-scraper-image:latest"
sudo docker run -d --name utr-scraper \
    -e UTR_EMAIL="$UTR_EMAIL" \
    -e UTR_PASSWORD="$UTR_PASSWORD" \
    -e GOOGLE_CLOUD_PROJECT="cpsc324-project-452600" \
    -e GCS_BUCKET_NAME="utr_scraper_bucket" \
    us-west1-docker.pkg.dev/cpsc324-project-452600/utr-scraper-repo/utr-scraper-image:latest

# Output container logs
echo "Container started. To view logs, run: sudo docker logs utr-scraper"

# Start monitoring loop to check for shutdown signal
echo "Starting shutdown signal monitor"

# Create a background job that checks for the shutdown signal
(
    # Loop until shutdown signal is found or 8 hours pass (as a safety measure)
    START_TIME=$(date +%s)
    MAX_RUNTIME=$((8 * 60 * 60))  # 8 hours in seconds
    
    while true; do
        # Check if the signal file exists
        if gsutil -q stat gs://utr_scraper_bucket/shutdown_signal.txt; then
            echo "Shutdown signal detected. Shutting down VM..."
            
            # Download the signal file to log its contents
            gsutil cp gs://utr_scraper_bucket/shutdown_signal.txt /tmp/shutdown_signal.txt
            echo "Signal file contents:"
            cat /tmp/shutdown_signal.txt
            
            # Delete the signal file so it doesn't affect future runs
            echo "Deleting shutdown signal file..."
            gsutil rm gs://utr_scraper_bucket/shutdown_signal.txt
            
            # Stop the container gracefully
            echo "Stopping Docker container..."
            sudo docker stop utr-scraper
            
            # Shutdown the VM
            echo "Shutting down VM in 30 seconds..."
            sudo shutdown -h +1
            
            # Exit the loop
            break
        fi
        
        # Check if max runtime exceeded
        CURRENT_TIME=$(date +%s)
        ELAPSED_TIME=$((CURRENT_TIME - START_TIME))
        if [ $ELAPSED_TIME -gt $MAX_RUNTIME ]; then
            echo "Maximum runtime of 8 hours exceeded. Shutting down VM..."
            sudo shutdown -h +1
            break
        fi
        
        # Wait for 2 minutes before checking again
        sleep 120
    done
) &

# Log that monitoring has started
echo "Shutdown signal monitor started in background" 