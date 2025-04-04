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

def upload_to_gcs(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        blob = bucket.blob(destination_blob_name)
        logger.info(f"Starting upload of {source_file_name} to {BUCKET_NAME}")
        blob.upload_from_filename(source_file_name)
        logger.info(f"Successfully uploaded {source_file_name} to {BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        save_logs_to_gcs(f"Error uploading file: {str(e)}")

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

# Function kept for compatibility but now it just logs instead of stopping VM
def stop_instance():
    """Previously stopped the VM, now just logs for debugging."""
    try:
        logger.info("VM stopping functionality disabled for debugging")
        save_logs_to_gcs("VM stopping functionality disabled for debugging")
    except Exception as e:
        logger.error(f"Error in stop_instance placeholder: {str(e)}")
        save_logs_to_gcs(f"Error in stop_instance placeholder: {str(e)}")

# Start execution
start_time = time.time()
logger.info("Starting UTR scraper...")
logger.info("Script version: 1.0.2 - GCP Debug Execution (VM stop disabled)")
save_logs_to_gcs("Starting UTR scraper on GCP... (VM stop disabled)")

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
        # Process all profiles in one go
        logger.info(f"Processing {len(profile_ids)} profiles")
        save_logs_to_gcs(f"Processing {len(profile_ids)} profiles")
        
        # Scrape UTR history for all profiles
        scrape_utr_history(profile_ids, email, password, 
                          offset=0, stop=-1, writer=writer)
        
        # Upload results
        try:
            upload_to_gcs(output_file, 'utr_history.csv')
            logger.info("Successfully uploaded results")
            save_logs_to_gcs("Successfully uploaded results")
        except Exception as e:
            logger.error(f"Error uploading results: {str(e)}")
            save_logs_to_gcs(f"Error uploading results: {str(e)}")
        
        logger.info("Scraping completed successfully")
        save_logs_to_gcs("Scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        save_logs_to_gcs(f"Error during scraping: {str(e)}")
        # Try to upload partial results
        try:
            upload_to_gcs(output_file, 'utr_history.csv')
            logger.info("Uploaded partial results")
            save_logs_to_gcs("Uploaded partial results")
        except Exception as upload_error:
            logger.error(f"Error uploading partial results: {str(upload_error)}")
            save_logs_to_gcs(f"Error uploading partial results: {str(upload_error)}")
        raise
    finally:
        # Log completion but don't stop the VM
        logger.info("Execution finished, VM stopping disabled for debugging")
        save_logs_to_gcs("Execution finished, VM stopping disabled for debugging")
        # Commented out the stop_instance call
        # stop_instance()

logger.info("Script execution complete, VM will continue running for debugging")
save_logs_to_gcs("Script execution complete, VM will continue running for debugging") 