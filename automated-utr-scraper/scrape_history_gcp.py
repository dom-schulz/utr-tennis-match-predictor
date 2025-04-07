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
import traceback
from collections import deque
import threading
import requests
import socket

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "utr_scraper_bucket")
UPLOAD_FILE_NAME = "utr_history.csv"  # file to upload to GCS after scraping
LOCAL_PROFILE_FILE = "profile_id.csv"  # profile file bundled with the Docker image

# Get credentials from environment variables, secrets passed in as environment 
# variables via built in functionality in GCP
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

# Log buffer/limit (prevents errors in logging and missing statements)
log_buffer = deque(maxlen=100)  # Store up to 100 log messages
log_buffer_lock = threading.Lock()
last_log_upload_time = time.time()
LOG_UPLOAD_INTERVAL = 5  # Upload logs every 5 seconds

def upload_to_gcs(source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        # First verify the file exists and has content
        if not os.path.exists(source_file_name):
            logger.error(f"File {source_file_name} does not exist")
            save_logs_to_gcs(f"Error: File {source_file_name} does not exist")
            return False
            
        file_size = os.path.getsize(source_file_name)
        if file_size == 0:
            logger.error(f"File {source_file_name} is empty (0 bytes)")
            save_logs_to_gcs(f"Error: File {source_file_name} is empty (0 bytes)")
            return False
            
        logger.info(f"Starting upload of {source_file_name} ({file_size} bytes) to {BUCKET_NAME}")
        
        # Log file content for debugging
        try:
            with open(source_file_name, 'r') as f:
                content_sample = f.read(1000)  # Read first 1000 chars
                lines = content_sample.count('\n') + 1
                logger.info(f"File content sample (first {lines} lines): {content_sample[:200]}...")
                save_logs_to_gcs(f"File content sample: {lines} lines, starts with: {content_sample[:100]}...")
        except Exception as e:
            logger.warning(f"Could not read file content: {str(e)}")
            
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logger.info(f"Successfully uploaded {source_file_name} to {BUCKET_NAME}/{destination_blob_name}")
        return True
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        logger.error(traceback.format_exc())
        save_logs_to_gcs(f"Error uploading file: {str(e)}")
        return False

def save_logs_to_gcs(log_message):
    """Buffers log messages and periodically writes them to GCS to avoid rate limits."""
    global last_log_upload_time # tracker for last time logs were uploaded
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {log_message}"
    
    with log_buffer_lock:
        # Add the message to our buffer
        log_buffer.append(formatted_message)
        
        # Only upload logs if enough time has passed since the last upload
        current_time = time.time()
        if current_time - last_log_upload_time >= LOG_UPLOAD_INTERVAL:
            try:
                # Get the log file or create a new one
                log_blob = bucket.blob('logs/scraper_log.txt')
                
                try:
                    # Try to download existing log content
                    current_log = log_blob.download_as_text()
                except Exception:
                    current_log = ""
                
                # Add all buffered logs
                buffered_logs = "\n".join(log_buffer)
                updated_log = f"{current_log}\n{buffered_logs}" if current_log else buffered_logs
                
                # Upload updated log
                log_blob.upload_from_string(updated_log)
                logger.info(f"Uploaded {len(log_buffer)} log messages to GCS")
                
                # Clear the buffer after successful upload
                log_buffer.clear()
                
                # Update the last upload time
                last_log_upload_time = current_time
            except Exception as e:
                logger.error(f"Error saving logs to GCS: {str(e)}")
                # Don't clear buffer on error - we'll try again later

def flush_logs():
    """Force all buffered logs to be written to GCS."""
    global last_log_upload_time
    
    with log_buffer_lock:
        if not log_buffer:
            return  # Nothing to flush
            
        try:
            # Get the log file or create a new one
            log_blob = bucket.blob('logs/scraper_log.txt')
            
            try:
                # Try to download existing log content
                current_log = log_blob.download_as_text()
            except Exception:
                current_log = ""
            
            # Add all buffered logs
            buffered_logs = "\n".join(log_buffer)
            updated_log = f"{current_log}\n{buffered_logs}" if current_log else buffered_logs
            
            # Upload updated log
            log_blob.upload_from_string(updated_log)
            logger.info(f"Flushed {len(log_buffer)} log messages to GCS")
            
            # Clear the buffer after successful upload
            log_buffer.clear()
            
            # Update the last upload time
            last_log_upload_time = time.time()
        except Exception as e:
            logger.error(f"Error flushing logs to GCS: {str(e)}")

def shutdown_vm():
    """Shut down the VM instance after the job completes."""
    try:
        # First try to get instance name from metadata server
        logger.info("Attempting to shut down VM...")
        save_logs_to_gcs("Attempting to shut down VM...")

        # Get instance information from metadata server
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/"
        headers = {"Metadata-Flavor": "Google"}
        
        # Get instance name and zone
        instance_name = requests.get(metadata_url + "name", headers=headers).text
        zone_url = metadata_url + "zone"
        zone_path = requests.get(zone_url, headers=headers).text
        project_id = requests.get(metadata_url + "project/project-id", headers=headers).text
        
        # Extract just the zone name from the path
        zone = zone_path.split('/')[-1]
        
        logger.info(f"Shutting down VM: {instance_name} in zone {zone}, project {project_id}")
        save_logs_to_gcs(f"Shutting down VM: {instance_name} in zone {zone}, project {project_id}")
        
        # Initialize the Compute Engine API client
        instances_client = compute_v1.InstancesClient()
        
        # Create the stop request
        operation = instances_client.stop(
            project=project_id,
            zone=zone,
            instance=instance_name
        )
        
        logger.info(f"Stop operation initiated: {operation.name}")
        save_logs_to_gcs(f"Stop operation initiated: {operation.name}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in shutdown_vm: {str(e)}")
        logger.error(traceback.format_exc())
        save_logs_to_gcs(f"Error in shutdown_vm: {str(e)}")
        return False



########## Run the scraper ##########
start_time = time.time()
logger.info("Starting UTR scraper...")
logger.info("Script version: 1.0.4 - Auto Shutdown VM")
save_logs_to_gcs("Starting UTR scraper on GCP with auto shutdown capability...")

# Save environment variables to log for debugging
env_vars = {k: v for k, v in os.environ.items() if 'UTR' in k or 'GCS' in k}
logger.info(f"Environment variables: {env_vars}")

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
    
    logger.info(f"Found profile file: {LOCAL_PROFILE_FILE}")
    file_size = os.path.getsize(LOCAL_PROFILE_FILE)
    logger.info(f"Profile file size: {file_size} bytes")
    
    # # Log file content for debugging
    # with open(LOCAL_PROFILE_FILE, 'r') as f:
    #     content = f.read()
    #     logger.info(f"Profile file content:\n{content}")
    #     save_logs_to_gcs(f"Profile file contains {content.count(chr(10))+1} lines")
        
    # Read the CSV file
    profile_ids = pd.read_csv(LOCAL_PROFILE_FILE)
    
    # Convert p_id column to integer, handling NaN designation
    if 'p_id' in profile_ids.columns:
        # First remove any rows with NaN or empty values in p_id
        before_count = len(profile_ids)
        profile_ids = profile_ids.dropna(subset=['p_id'])
        dropped_count = before_count - len(profile_ids)
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with missing p_id values")
            save_logs_to_gcs(f"Dropped {dropped_count} rows with missing p_id values")
            
        # Then convert to integer
        try:
            profile_ids['p_id'] = profile_ids['p_id'].astype(int)
            logger.info(f"Converted p_id column to integer type")
            save_logs_to_gcs(f"Converted p_id column to integer type")
        except Exception as e:
            logger.error(f"Error converting p_id to integer: {str(e)}")
            logger.error(profile_ids['p_id'].to_list())
            save_logs_to_gcs(f"Error converting p_id to integer: {str(e)}")
    else:
        logger.error("p_id column not found in profile file")
        save_logs_to_gcs("p_id column not found in profile file")
        exit(1)
    
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
    logger.error(traceback.format_exc())
    save_logs_to_gcs(f"Error reading profile CSV file: {str(e)}")
    exit(1)

# Create output file
output_file = 'utr_history.csv'

# Scrape UTR history for all profiles and get the resulting dataframe
logger.info(f"Processing {len(profile_ids)} profiles")
save_logs_to_gcs(f"Processing {len(profile_ids)} profiles")

try:
    # Set stop=-1 to process all profiles (ie. no limit)
    results_df = scrape_utr_history(profile_ids, email, password, offset=0, stop=-1, writer=None)
    
    if results_df is None or len(results_df) == 0:
        logger.error("Scraping returned empty results")
        save_logs_to_gcs("Error: Scraping returned empty results")
        
        # Create a dummy record for debugging
        logger.info("Creating dummy record for debugging")
        dummy_data = {
            'first_name': ['DEBUG'], 
            'last_name': ['RECORD'],
            'date': [datetime.now().strftime('%Y-%m-%d')],
            'utr': ['0.0']
        }
        results_df = pd.DataFrame(dummy_data)
        save_logs_to_gcs("Created dummy record for debugging purposes")
    
    # Log the number of records found
    logger.info(f"Scraping completed. Found {len(results_df)} UTR records")
    save_logs_to_gcs(f"Scraping completed. Found {len(results_df)} UTR records")
    
    # Save dataframe to local CSV file
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(results_df)} records to local file {output_file}")
    
    # Verify the file was created and has content
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        logger.info(f"Output file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error("Output file is empty (0 bytes)")
            save_logs_to_gcs("Error: Output file is empty (0 bytes)")
    else:
        logger.error(f"Output file {output_file} was not created")
        save_logs_to_gcs(f"Error: Output file {output_file} was not created")
    
    # Upload the CSV file to GCS bucket
    upload_success = upload_to_gcs(output_file, UPLOAD_FILE_NAME)
    if upload_success:
        logger.info(f"Successfully uploaded {output_file} to {BUCKET_NAME}/{UPLOAD_FILE_NAME}")
        save_logs_to_gcs(f"Successfully uploaded {len(results_df)} records to {BUCKET_NAME}/{UPLOAD_FILE_NAME}")
    else:
        logger.error("Failed to upload results to GCS")
        save_logs_to_gcs("Failed to upload results to GCS")
    
except Exception as e:
    logger.error(f"Error in scraping or upload process: {str(e)}")
    logger.error(traceback.format_exc())
    save_logs_to_gcs(f"Error in scraping or upload process: {str(e)}")
    
# Calculate execution time
execution_time = time.time() - start_time
logger.info(f"Script execution complete. Total time: {execution_time:.2f} seconds")
save_logs_to_gcs(f"Script execution complete. Total time: {execution_time:.2f} seconds")

# Shut down the VM
logger.info("Job complete, shutting down VM...")
save_logs_to_gcs("Job complete, shutting down VM...")
shutdown_success = shutdown_vm()

if shutdown_success:
    logger.info("VM shutdown initiated successfully")
    save_logs_to_gcs("VM shutdown initiated successfully")
else:
    logger.error("Failed to shut down VM")
    save_logs_to_gcs("Failed to shut down VM") 