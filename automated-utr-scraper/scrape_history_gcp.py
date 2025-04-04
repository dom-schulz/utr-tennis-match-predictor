from scraper import *
import pandas as pd
from google.cloud import storage
import csv
import io
import os
from selenium import webdriver
import logging
import time
from google.cloud import compute_v1
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "utr_scraper_bucket")
UPLOAD_FILE_NAME = "utr_history.csv"  # file to upload to GCS after scraping
LOCAL_PROFILE_FILE = "profile_id.csv"  # profile file bundled with the Docker image

# Get credentials from environment variables, secrets passed in as environment 
# variables via built in functionality in Cloud Run
email = os.getenv("UTR_EMAIL")
password = os.getenv("UTR_PASSWORD")

# Initialize GCS client
client = storage.Client() 
bucket = client.bucket(BUCKET_NAME)
upload_blob = bucket.blob(UPLOAD_FILE_NAME)

# Create and initialize StringIO object to write CSV data
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer) # take file like object (csv_buffer) and prepares it for writing
writer.writerow(['f_name', 'l_name', 'date', 'utr']) # write headers to csv

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    logger.info(f"Starting upload of {source_file_name} to {bucket_name}")
    blob.upload_from_filename(source_file_name)
    logger.info(f"Successfully uploaded {source_file_name} to {bucket_name}")

def save_logs_to_gcs(log_message):
    """Saves log messages to a file in GCS."""
    try:
        # Get the current log file or create a new one
        log_blob = bucket.blob('logs/scraper_log.txt')
        
        # Try to download existing log content
        try:
            current_log = log_blob.download_as_text()
        except Exception:
            current_log = ""
        
        # Append new log entry with timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        updated_log = f"{current_log}\n[{timestamp}] {log_message}"
        
        # Upload updated log
        log_blob.upload_from_string(updated_log)
        logger.info(f"Log saved to GCS: {log_message}")
    except Exception as e:
        logger.error(f"Error saving log to GCS: {str(e)}")

def stop_instance():
    """Stops the current Compute Engine instance."""
    try:
        # Get instance metadata
        metadata_client = compute_v1.InstancesClient()
        project = 'cpsc324-project-452600'
        zone = 'us-west1-a'
        instance_name = 'utr-scraper-vm'
        
        logger.info("Stopping Compute Engine instance...")
        save_logs_to_gcs("Stopping Compute Engine instance...")
        operation = metadata_client.stop(
            project=project,
            zone=zone,
            instance=instance_name
        )
        operation.result()  # Wait for the operation to complete
        logger.info("Instance stopped successfully")
        save_logs_to_gcs("Instance stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping instance: {str(e)}")
        save_logs_to_gcs(f"Error stopping instance: {str(e)}")

# Start execution
start_time = time.time()
logger.info("Starting UTR scraper...")
logger.info("Script version: 1.0.1 - GCP Execution")
save_logs_to_gcs("Starting UTR scraper on GCP...")

# Get credentials from environment variables
email = os.environ.get('UTR_EMAIL')
password = os.environ.get('UTR_PASSWORD')

logger.info(f"Environment variables - Email set: {email is not None}, Password set: {password is not None}")
save_logs_to_gcs(f"Environment variables - Email set: {email is not None}, Password set: {password is not None}")

if not email or not password:
    logger.error("UTR credentials not found in environment variables")
    save_logs_to_gcs("UTR credentials not found in environment variables")
    exit(1)

# Read the local CSV file bundled with the Docker image
try:
    # Check if file exists
    if not os.path.exists(LOCAL_PROFILE_FILE):
        logger.error(f"Profile file {LOCAL_PROFILE_FILE} not found in Docker image")
        save_logs_to_gcs(f"Profile file {LOCAL_PROFILE_FILE} not found in Docker image")
        exit(1)
        
    profile_ids = pd.read_csv(LOCAL_PROFILE_FILE)
    logger.info(f"Successfully read {len(profile_ids)} profiles from local file")
    save_logs_to_gcs(f"Successfully read {len(profile_ids)} profiles from local file")
    
    # Check and rename columns if needed
    if 'p_id' in profile_ids.columns and 'profile_id' not in profile_ids.columns:
        # Make a copy of the p_id column as profile_id for compatibility
        profile_ids['profile_id'] = profile_ids['p_id']
        logger.info("Added profile_id column based on p_id for compatibility")
        save_logs_to_gcs("Added profile_id column based on p_id for compatibility")
    
    if 'f_name' in profile_ids.columns and 'first_name' not in profile_ids.columns:
        # Make a copy of the f_name column as first_name for compatibility
        profile_ids['first_name'] = profile_ids['f_name']
        logger.info("Added first_name column based on f_name for compatibility")
        save_logs_to_gcs("Added first_name column based on f_name for compatibility")
    
    if 'l_name' in profile_ids.columns and 'last_name' not in profile_ids.columns:
        # Make a copy of the l_name column as last_name for compatibility
        profile_ids['last_name'] = profile_ids['l_name']
        logger.info("Added last_name column based on l_name for compatibility")
        save_logs_to_gcs("Added last_name column based on l_name for compatibility")
    
except Exception as e:
    logger.error(f"Error reading profile CSV file: {str(e)}")
    save_logs_to_gcs(f"Error reading profile CSV file: {str(e)}")
    exit(1)

# Create output file
output_file = 'utr_history.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['First Name', 'Last Name', 'Date', 'UTR'])
    
    try:
        # Process profiles in smaller batches
        batch_size = 10
        total_profiles = len(profile_ids)
        
        for i in range(0, total_profiles, batch_size):
            batch_end = min(i + batch_size, total_profiles)
            logger.info(f"Processing profiles {i+1} to {batch_end} of {total_profiles}")
            save_logs_to_gcs(f"Processing profiles {i+1} to {batch_end} of {total_profiles}")
            
            # Check if we're approaching the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > 3540:  # 59 minutes
                logger.warning("Approaching timeout, saving progress...")
                save_logs_to_gcs("Approaching timeout, saving progress...")
                break
            
            scrape_utr_history(profile_ids.iloc[i:batch_end], email, password, 
                             offset=0, stop=-1, writer=writer)
            
            # Upload progress after each batch
            try:
                upload_to_gcs(bucket_name, output_file, 'utr_history.csv')
                logger.info(f"Successfully uploaded batch {i//batch_size + 1}")
                save_logs_to_gcs(f"Successfully uploaded batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error uploading batch: {str(e)}")
                save_logs_to_gcs(f"Error uploading batch: {str(e)}")
        
        logger.info("Scraping completed successfully")
        save_logs_to_gcs("Scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        save_logs_to_gcs(f"Error during scraping: {str(e)}")
        # Try to upload partial results
        try:
            upload_to_gcs(bucket_name, output_file, 'utr_history.csv')
            logger.info("Uploaded partial results")
            save_logs_to_gcs("Uploaded partial results")
        except Exception as upload_error:
            logger.error(f"Error uploading partial results: {str(upload_error)}")
            save_logs_to_gcs(f"Error uploading partial results: {str(upload_error)}")
        raise
    finally:
        # Stop the instance after scraping is complete
        logger.info("Execution finished, stopping VM instance")
        save_logs_to_gcs("Execution finished, stopping VM instance")
        stop_instance() 