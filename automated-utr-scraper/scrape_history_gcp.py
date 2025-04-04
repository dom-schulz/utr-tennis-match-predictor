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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "utr_scraper_bucket")
UPLOAD_FILE_NAME = "utr_history.csv"  # file to upload to GCS after scraping
DOWNLOAD_FILE_NAME = "profile_id.csv"  # file to download from GCS before scraping

# Get credentials from environment variables, secrets passed in as environment 
# variables via built in functionality in Cloud Run
email = os.getenv("UTR_EMAIL")
password = os.getenv("UTR_PASSWORD")

# Initialize GCS client
client = storage.Client() 
bucket = client.bucket(BUCKET_NAME)
upload_blob = bucket.blob(UPLOAD_FILE_NAME)
download_blob = bucket.blob(DOWNLOAD_FILE_NAME)

# Create and initialize StringIO object to write CSV data
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer) # take file like object (csv_buffer) and prepares it for writing
writer.writerow(['f_name', 'l_name', 'date', 'utr']) # write headers to csv

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} from {bucket_name}")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    logger.info(f"Starting upload of {source_file_name} to {bucket_name}")
    blob.upload_from_filename(source_file_name)
    logger.info(f"Successfully uploaded {source_file_name} to {bucket_name}")

def stop_instance():
    """Stops the current Compute Engine instance."""
    try:
        # Get instance metadata
        metadata_client = compute_v1.InstancesClient()
        project = 'cpsc324-project-452600'
        zone = 'us-west1-a'
        instance_name = 'utr-scraper-vm'
        
        logger.info("Stopping Compute Engine instance...")
        operation = metadata_client.stop(
            project=project,
            zone=zone,
            instance=instance_name
        )
        operation.result()  # Wait for the operation to complete
        logger.info("Instance stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping instance: {str(e)}")

# Start execution
start_time = time.time()
logger.info("Starting UTR scraper...")
logger.info("Script version: 1.0.1 - Direct Execution")

# Get credentials from environment variables
email = os.environ.get('UTR_EMAIL')
password = os.environ.get('UTR_PASSWORD')

logger.info(f"Environment variables - Email set: {email is not None}, Password set: {password is not None}")

if not email or not password:
    logger.error("UTR credentials not found in environment variables")
    exit(1)

# Download profile_ids.csv from GCS
bucket_name = 'utr_scraper_bucket'
source_blob_name = 'profile_id.csv'
destination_file_name = 'profile_id.csv'

try:
    download_from_gcs(bucket_name, source_blob_name, destination_file_name)
    logger.info("Successfully downloaded profile_ids.csv")
except Exception as e:
    logger.error(f"Error downloading profile_ids.csv: {str(e)}")
    exit(1)

# Read the CSV file
try:
    profile_ids = pd.read_csv(destination_file_name)
    logger.info(f"Successfully read {len(profile_ids)} profiles")
except Exception as e:
    logger.error(f"Error reading profile_ids.csv: {str(e)}")
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
            
            # Check if we're approaching the timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > 3540:  # 59 minutes
                logger.warning("Approaching timeout, saving progress...")
                break
            
            scrape_utr_history(profile_ids.iloc[i:batch_end], email, password, 
                             offset=0, stop=-1, writer=writer)
            
            # Upload progress after each batch
            try:
                upload_to_gcs(bucket_name, output_file, 'utr_history.csv')
                logger.info(f"Successfully uploaded batch {i//batch_size + 1}")
            except Exception as e:
                logger.error(f"Error uploading batch: {str(e)}")
        
        logger.info("Scraping completed successfully")
        
    except Exception as e:
        logger.error(f"Error during scraping: {str(e)}")
        # Try to upload partial results
        try:
            upload_to_gcs(bucket_name, output_file, 'utr_history.csv')
            logger.info("Uploaded partial results")
        except Exception as upload_error:
            logger.error(f"Error uploading partial results: {str(upload_error)}")
        raise
    finally:
        # Stop the instance after scraping is complete
        stop_instance() 