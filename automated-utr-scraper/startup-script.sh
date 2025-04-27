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
    us-west1-docker.pkg.dev/cpsc324-project-452600/utr-scraper-repo/utr-scraper-image:latest

# Output container logs
echo "Container started. To view logs, run: sudo docker logs utr-scraper"

# Create a monitoring script and write it to a file
cat > /tmp/monitor_container.sh << 'EOF'
#!/bin/bash

# Function to log messages
log_message() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> /tmp/monitor.log
}

# Get current instance information
INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | cut -d/ -f4)
PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")

log_message "Starting monitor for container utr-scraper on instance $INSTANCE_NAME in zone $ZONE"

# Default sleep time in seconds
SLEEP_TIME=60
CONTAINER_NAME="utr-scraper"
MAX_RUNTIME_HOURS=10  # Safety measure: max runtime 10 hours

# Get start time in seconds
START_TIME=$(date +%s)
MAX_RUNTIME_SECONDS=$((MAX_RUNTIME_HOURS * 3600))

# Monitor the container
while true; do
  # Check if container is still running
  CONTAINER_STATUS=$(sudo docker inspect -f '{{.State.Status}}' $CONTAINER_NAME 2>/dev/null || echo "not found")
  CONTAINER_RUNNING=$(sudo docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null || echo "false")
  
  # Log current status
  log_message "Container status: $CONTAINER_STATUS, Running: $CONTAINER_RUNNING"
  
  # Calculate elapsed time
  CURRENT_TIME=$(date +%s)
  ELAPSED_SECONDS=$((CURRENT_TIME - START_TIME))
  ELAPSED_HOURS=$((ELAPSED_SECONDS / 3600))
  ELAPSED_MINUTES=$(((ELAPSED_SECONDS % 3600) / 60))
  
  log_message "Elapsed time: ${ELAPSED_HOURS}h ${ELAPSED_MINUTES}m"
  
  # Check if container has exited
  if [ "$CONTAINER_STATUS" = "exited" ] || [ "$CONTAINER_STATUS" = "not found" ]; then
    log_message "Container has exited or not found, checking exit code..."
    
    # Get exit code if container exists
    if [ "$CONTAINER_STATUS" != "not found" ]; then
      EXIT_CODE=$(sudo docker inspect -f '{{.State.ExitCode}}' $CONTAINER_NAME)
      log_message "Container exited with code: $EXIT_CODE"
      
      # If container exited with success (code 0)
      if [ "$EXIT_CODE" = "0" ]; then
        log_message "Container completed successfully. Shutting down VM..."
        sudo gsutil cp /tmp/monitor.log gs://utr_scraper_bucket/logs/monitor.log
        sudo gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
        exit 0
      else
        log_message "Container exited with error. Not shutting down VM to allow debugging."
        exit 1
      fi
    else
      log_message "Container not found. Shutting down VM..."
      sudo gsutil cp /tmp/monitor.log gs://utr_scraper_bucket/logs/monitor.log
      sudo gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
      exit 0
    fi
  fi
  
  # Check if we've exceeded max runtime (safety measure)
  if [ $ELAPSED_SECONDS -gt $MAX_RUNTIME_SECONDS ]; then
    log_message "Exceeded maximum runtime of $MAX_RUNTIME_HOURS hours. Shutting down VM..."
    sudo docker stop $CONTAINER_NAME
    sudo gsutil cp /tmp/monitor.log gs://utr_scraper_bucket/logs/monitor.log
    sudo gcloud compute instances stop $INSTANCE_NAME --zone=$ZONE --project=$PROJECT_ID
    exit 0
  fi
  
  # Wait before checking again
  sleep $SLEEP_TIME
done
EOF

# Make the script executable
chmod +x /tmp/monitor_container.sh

# Start the monitoring script in the background
nohup /tmp/monitor_container.sh > /tmp/monitor_output.log 2>&1 &
echo "Started container monitoring script (PID: $!)" 