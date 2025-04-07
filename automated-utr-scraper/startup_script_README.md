# Scraper VM Startup Script 

The [`startup-script.sh`](https://github.com/dom-schulz/utr-tennis-match-predictor/blob/main/automated-utr-scraper/statup-script.sh) is a bash script that handles the startup and shutdomw of the scraping process on GCP:

### 1. Initial Setup

- Installs Docker and all dependencies
- Installs Google Cloud SDK
- Configures Docker to use Google Cloud Registry
- Retrieves credentials securely from Secret Manager

### 2. Container Management

- Pulls the latest scraper Docker image from GCP Artifact Registry
- Stops and removes any existing containers with the same name
- Runs the scraper container with appropriate environment variables:
  - UTR credentials (email/password)
  - GCP project ID
  - GCS bucket name for data storage

### 3. Monitoring and Auto-Shutdown

The script implements a monitoring system that:

- Runs as a background process
- Checks container status every 60 seconds
- Logs container status, running state, and elapsed time
- Uploads monitoring logs to GCS bucket

#### Auto-Shutdown Logic

The VM automatically shuts down based on container status:

1. **Successful Completion**: If the container exits with code 0 (success), the VM is stopped
2. **Error State**: If the container exits with a non-zero code, the VM remains running for debugging
3. **Safety Timeout**: If the container runs longer than 5 hours (configurable), the VM is stopped

### 4. Logging

- All actions are logged to `/tmp/monitor.log` locally
- Logs are copied to GCS bucket `utr_scraper_bucket/logs/monitor.log` before shutdown
- Full monitoring output is also saved to `/tmp/monitor_output.log`

## Usage

1. Create a GCP VM instance with this script as the startup script
2. Ensure the VM has appropriate permissions:
   - Secret Manager access for credentials
   - GCS bucket access for data/log storage
   - Compute Engine access to stop itself

3. The VM will:
   - Start automatically (scheduled twice a week via Cloud Function)
   - Run the scraper (ie. collect data)
   - Write to the GCS bucket
   - Shut down upon completion
   - Upload logs for troubleshooting

## Environment Variables

The container expects these environment variables:

- `UTR_EMAIL`: UTR account email (from Secret Manager)
- `UTR_PASSWORD`: UTR account password (from Secret Manager)
- `GOOGLE_CLOUD_PROJECT`: The GCP project ID
- `GCS_BUCKET_NAME`: The GCS bucket to store scraped data and logs

## Safety Features

- Maximum runtime limit (set at 5 hours)
- Container status monitoring
- Error detection with VM preservation for debugging
- Comprehensive logging
- Automatic removal of stale containers

## Maintenance

1. If changing the monitoring parameters, update the MAX_RUNTIME_HOURS or SLEEP_TIME variables
