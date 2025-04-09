from scraper import scrape_player_matches
import pandas as pd
from google.cloud import storage
import csv
import io
import os
import logging
import traceback
from google.oauth2 import service_account

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get bucket names from environment variables
MATCHES_BUCKET_NAME = os.getenv("GCS_MATCHES_BUCKET_NAME", "matches-scraper-bucket")
UTR_BUCKET_NAME = os.getenv("GCS_UTR_BUCKET_NAME", "utr_scraper_bucket")

# GCS File Paths
UTR_HISTORY_FILE = "utr_history.csv"
MATCHES_OUTPUT_FILE = "atp_utr_tennis_matches.csv"
PROFILE_ID_FILE = "profile_id.csv"

# Get credentials from environment variables
email = os.getenv("UTR_EMAIL")
password = os.getenv("UTR_PASSWORD")

def download_csv_from_gcs(bucket, file_path):
    """Downloads a CSV from GCS and returns a pandas DataFrame."""
    try:
        blob = bucket.blob(file_path)
        data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data))
        logger.info(f"Successfully downloaded and read {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error downloading or reading CSV from GCS: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def upload_df_to_gcs(df, bucket, file_path):
    """Uploads a pandas DataFrame to GCS as a CSV."""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8')
        blob = bucket.blob(file_path)
        blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
        logger.info(f"Successfully uploaded {file_path} to GCS")
        return True
    except Exception as e:
        logger.error(f"Error uploading DataFrame to GCS: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_player_history(utr_history):
    history = {}
    for i in range(len(utr_history)):
        if utr_history['f_name'][i]+' '+utr_history['l_name'][i] not in history.keys():
            history[utr_history['f_name'][i]+' '+utr_history['l_name'][i]] = [[utr_history['utr'][i], utr_history['date'][i]]]
        else:
            history[utr_history['f_name'][i]+' '+utr_history['l_name'][i]].append([utr_history['utr'][i], utr_history['date'][i]])
    return history

try:
    # Initialize GCS client using default credentials for GCP or explicit file if provided
    # This handles both GCP VM (no explicit credentials needed) and local development
    logger.info("Initializing GCS client...")
    creds_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    
    if creds_file:
        # Use explicit credentials from file (for local development)
        logger.info(f"Using credentials from file: {creds_file}")
        credentials = service_account.Credentials.from_service_account_file(creds_file)
        client = storage.Client(credentials=credentials)
    else:
        # Use default credentials (for GCP VM/Cloud Run)
        logger.info("Using default GCP credentials")
        client = storage.Client()
    
    matches_bucket = client.bucket(MATCHES_BUCKET_NAME)
    utr_bucket = client.bucket(UTR_BUCKET_NAME)
    
    # Download required files from GCS (equivalent to original pd.read_csv calls)
    profile_ids = download_csv_from_gcs(utr_bucket, PROFILE_ID_FILE)
    utr_history = download_csv_from_gcs(utr_bucket, UTR_HISTORY_FILE)
    prev_matches = download_csv_from_gcs(matches_bucket, MATCHES_OUTPUT_FILE)

    # Process UTR history exactly as in original
    utr_history = get_player_history(utr_history)

    # Use StringIO to capture new matches data in memory (equivalent to original file writing)
    new_matches_buffer = io.StringIO()
    writer = csv.writer(new_matches_buffer)
    
    # Write the header row first
    writer.writerow(['date', 'p1', 'p2', 'p1_id', 'p2_id', 'p1_utr', 'p2_utr', 'tournament_category', 'score', 'winner'])

    # Run scraping exactly as in original
    scrape_player_matches(profile_ids, utr_history, prev_matches, email, password, offset=0, stop=-1, writer=writer)

    # Read the newly scraped matches and process exactly as in original
    new_matches_buffer.seek(0)
    matches = pd.read_csv(new_matches_buffer)
    
    # Log DataFrame info for debugging
    logger.info(f"DataFrame columns: {matches.columns.tolist()}")
    logger.info(f"DataFrame shape: {matches.shape}")
    
    if len(matches) > 0:
        # Check if required columns exist
        required_cols = ['date', 'p1', 'p2']
        missing_cols = [col for col in required_cols if col not in matches.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise KeyError(f"Missing required columns: {missing_cols}")
            
        matches.drop_duplicates(subset=['date','p1','p2'], inplace=True)
        
        # Upload to GCS (equivalent to original to_csv)
        upload_df_to_gcs(matches, matches_bucket, MATCHES_OUTPUT_FILE)
    else:
        logger.warning("No new matches found in the scraping process")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    logger.error(traceback.format_exc())
    raise