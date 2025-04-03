from scraper import *
import pandas as pd
from google.cloud import storage
import csv
import io
import os

# Set bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "utr_scraper_bucket")
UPLOAD_FILE_NAME = "utr_history.csv"  # file to upload to GCS after scraping
DOWNLOAD_FILE_NAME = "profile_id.csv"  # file to download from GCS before scraping

# Get credentials from environment variables
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

# Download profile_id.csv as a string and read it into a DataFrame
data = download_blob.download_as_string()
profile_ids = pd.read_csv(io.BytesIO(data)) # read string data into dataframe, it needs to be converted to bytes before reading

# Read profile_id.csv and scrape UTR history
scrape_utr_history(profile_ids, email, password, offset=0, stop=-1, writer=writer)

# Upload the CSV data to GCS
upload_blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
print(f"Successfully uploaded {UPLOAD_FILE_NAME} to bucket {BUCKET_NAME}") 
