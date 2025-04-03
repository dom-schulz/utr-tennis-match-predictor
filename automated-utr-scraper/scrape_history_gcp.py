from scraper import *
import pandas as pd
from google.cloud import storage
import csv
import io
import os
from selenium import webdriver
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "utr_scraper_bucket")
UPLOAD_FILE_NAME = "utr_history.csv"  # file to upload to GCS after scraping
DOWNLOAD_FILE_NAME = "profile_id.csv"  # file to download from GCS before scraping

# Get credentials from environment variables, secrets passed in as environment 
# variables via built in functionality in Cloud Run
# email = os.getenv("UTR_EMAIL")
# password = os.getenv("UTR_PASSWORD")

email = 'jz1352@gmail.com'
password = 'Mr.Milom0nst3r!%'

# Initialize GCS client
client = storage.Client() 
bucket = client.bucket(BUCKET_NAME)
upload_blob = bucket.blob(UPLOAD_FILE_NAME)
download_blob = bucket.blob(DOWNLOAD_FILE_NAME)

# Create and initialize StringIO object to write CSV data
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer) # take file like object (csv_buffer) and prepares it for writing
writer.writerow(['f_name', 'l_name', 'date', 'utr']) # write headers to csv

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    logger.info(f"Downloaded {source_blob_name} from {bucket_name} to {destination_file_name}.")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    logger.info(f"Uploaded {source_file_name} to {bucket_name} as {destination_blob_name}.")

def main():
    try:
        logger.info("Starting UTR scraper...")
        
        # Get credentials from environment variables
        email = os.environ.get('UTR_EMAIL')
        password = os.environ.get('UTR_PASSWORD')
        
        if not email or not password:
            logger.error("UTR credentials not found in environment variables")
            return
        
        logger.info("Credentials loaded successfully")
        
        # Download profile_ids.csv from GCS
        bucket_name = 'utr_scraper_bucket'
        download_blob(bucket_name, 'profile_id.csv', 'profile_id.csv')
        logger.info("Profile IDs downloaded successfully")
        
        # Read profile IDs
        profile_ids = pd.read_csv('profile_id.csv')
        logger.info(f"Loaded {len(profile_ids)} profile IDs")
        
        # Create output file
        output_file = 'utr_history.csv'
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['First Name', 'Last Name', 'Date', 'UTR'])
            
            logger.info("Starting to scrape UTR history...")
            try:
                scrape_utr_history(profile_ids, email, password, offset=0, stop=-1, writer=writer)
                logger.info("UTR history scraping completed successfully")
            except Exception as e:
                logger.error(f"Error during scraping: {str(e)}")
                raise
        
        # Upload results to GCS
        upload_blob(bucket_name, output_file, 'utr_history.csv')
        logger.info("Results uploaded to GCS successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
