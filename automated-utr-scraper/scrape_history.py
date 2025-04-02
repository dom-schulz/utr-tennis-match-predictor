from scraper import *
import pandas as pd
from google.cloud import storage, secretmanager
import csv
import io
import os

def get_secret(secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/cpsc324-project-452600/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Set bucket name from environment variable
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "utr_scraper_bucket")
FILE_NAME = "utr_history.csv"

# Get credentials from Secret Manager
email = get_secret("utr-email")
password = get_secret("utr-password")

# Initialize GCS client
client = storage.Client()
bucket = client.bucket(BUCKET_NAME)
blob = bucket.blob(FILE_NAME)

# Create a StringIO object to write CSV data
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(['f_name', 'l_name', 'date', 'utr'])

profile_ids = pd.read_csv('profile_id.csv')
scrape_utr_history(profile_ids, email, password, offset=0, stop=-1, writer=writer)

# Upload the CSV data to GCS
blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
print(f"Successfully uploaded {FILE_NAME} to bucket {BUCKET_NAME}") 