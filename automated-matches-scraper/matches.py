from scraper import scrape_player_matches
import pandas as pd
from google.cloud import storage
import csv
import io
import os
import logging
import traceback
from google.oauth2 import service_account
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

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
        if utr_history['first_name'][i]+' '+utr_history['last_name'][i] not in history.keys():
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]] = [[utr_history['utr'][i], utr_history['date'][i]]]
        else:
            history[utr_history['first_name'][i]+' '+utr_history['last_name'][i]].append([utr_history['utr'][i], utr_history['date'][i]])

    return history

try:
    # # Initialize GCS client using default credentials for GCP or explicit file if provided
    logger.info("Initializing GCS client...")
    
    # client = storage.Client.from_service_account_json("credentials.json")
    
    # if client:
    #     # Use explicit credentials from file (for local development)
    #     logger.info(f"Using credentials from file: 'credentials.json'")
    
    # Use default credentials (for GCP VM)
    logger.info("Using default GCP credentials")
    client = storage.Client()
    
    matches_bucket = client.bucket(MATCHES_BUCKET_NAME)
    utr_bucket = client.bucket(UTR_BUCKET_NAME)
    
    # Download required files from GCS (equivalent to original pd.read_csv calls)
    profile_ids = download_csv_from_gcs(utr_bucket, PROFILE_ID_FILE)
    utr_history = download_csv_from_gcs(utr_bucket, UTR_HISTORY_FILE)
    prev_matches = download_csv_from_gcs(matches_bucket, MATCHES_OUTPUT_FILE)
    
    # print row count of three read in dfs 
    logger.info(f"Profile IDs: {len(profile_ids)}")
    logger.info(f"UTR History: {len(utr_history)}")
    logger.info(f"Previous Matches: {len(prev_matches)}")
    
    # Ensure all profile ids are in utr_history (must check first and last name existence)
    profile_ids = profile_ids[profile_ids['f_name'].isin(utr_history['first_name']) & profile_ids['l_name'].isin(utr_history['last_name'])]
    
    # Ensure all names in prev_matches (cols p1 and p2) are in utr_history (cols f_name and l_name). 
    # p1 and p2 are strings with full names, f name and l name are separated by a space    
    # Create full name column once
    utr_history['full_name'] = utr_history['first_name'].str.strip() + ' ' + utr_history['last_name'].str.strip()

    # Filter prev_matches by whether p1 and p2 are in full_name list
    prev_matches = prev_matches[
        prev_matches['p1'].isin(utr_history['full_name']) &
        prev_matches['p2'].isin(utr_history['full_name'])
    ]

    # print length of prev_matches
    logger.info(f"Prev Matches post filtering: {len(prev_matches)}")
    
    # Process UTR history exactly as in original
    utr_history = get_player_history(utr_history)

    # Use StringIO to capture new matches data in memory (equivalent to original file writing)
    new_matches_buffer = io.StringIO()
    writer = csv.writer(new_matches_buffer)
    
    # Write the header row first
    writer.writerow(['tournament','date','series','court','surface','round','best_of','p1','p1_utr','p2','p2_utr','winner','p1_games','p2_games','score','p_win'])

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
        required_cols = ['date', 'p1', 'p2', 'winner']
        missing_cols = [col for col in required_cols if col not in matches.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise KeyError(f"Missing required columns: {missing_cols}")
            
        matches.drop_duplicates(subset=['date','p1','p2','winner'], inplace=True)
        
        # print prev matches columns and matches columns, and row count
        logger.info(f"Prev Matches Columns: {prev_matches.columns.tolist()}")
        logger.info(f"Matches Columns: {matches.columns.tolist()}")
        logger.info(f"Prev Matches Row Count: {len(prev_matches)}")
        logger.info(f"Matches Row Count: {len(matches)}")
                
        # gcs write matches and prev_matches to csv
        upload_df_to_gcs(matches, matches_bucket, "matches.csv")
        upload_df_to_gcs(prev_matches, matches_bucket, "prev_matches.csv")
        
        ##### Make sure prev_matches rows are not already in matches
        
        # Convert prev_matches to list of its rows
        prev_match_rows = prev_matches.values.tolist()
        initial_prev_match_row_count = len(prev_match_rows)

        # initialize append matches as empty dataframe with same columns as matches
        append_matches = pd.DataFrame(columns=matches.columns)

        for index, row in matches.iterrows():
            row_as_list = row.values.tolist() # convert current row to list

            if row_as_list not in prev_match_rows:
                # logger.info(f"Appending Match: {row_as_list}")
                prev_matches = pd.concat([prev_matches, pd.DataFrame([row])], ignore_index=True)
                
        # Upload to GCS (equivalent to original to_csv)
        upload_df_to_gcs(prev_matches, matches_bucket, MATCHES_OUTPUT_FILE)
        
        logger.info(f"Matches added: {len(prev_matches)-initial_prev_match_row_count}")
    else:
        logger.warning("No new matches found in the scraping process")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    logger.error(traceback.format_exc())
    raise