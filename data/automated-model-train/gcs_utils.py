import pandas as pd
import io

def download_csv_from_gcs(log_traceback, bucket, file_path):
    """Downloads a CSV from GCS and returns a pandas DataFrame."""
    
    logger = log_traceback[0]
    traceback = log_traceback[1]
    
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
    
def upload_model_to_gcs(log_traceback, model_binary, bucket, file_path):
    """Uploads a binary model object to GCS."""
    
    logger = log_traceback[0]
    traceback = log_traceback[1]
    
    try:
        blob = bucket.blob(file_path)
        blob.upload_from_string(model_binary, content_type='application/octet-stream')
        logger.info(f"Successfully uploaded binary model to {file_path} in GCS")
        return True
    except Exception as e:
        logger.error(f"Error uploading binary model to GCS: {str(e)}")
        logger.error(traceback.format_exc())
        return False
