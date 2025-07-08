import os
from google.cloud import secretmanager

def get_secret(secret_id: str, version_id: str = "latest") -> str:
    """
    Retrieves a secret from Google Cloud Secret Manager.

    This function authenticates using Application Default Credentials (ADC).
    For local development, run 'gcloud auth application-default login'.
    In a GCP environment (like Cloud Run), the service account associated
    with the resource will be used automatically.

    Args:
        secret_id: The ID of the secret to retrieve.
        version_id: The version of the secret (defaults to "latest").

    Returns:
        The secret value as a string.
    """
    project_id = os.environ.get("GCP_PROJECT_ID")
    if not project_id:
        # Try to get project ID from gcloud config
        try:
            import subprocess
            project_id = subprocess.check_output(
                ["gcloud", "config", "get-value", "project"],
                text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ValueError("GCP_PROJECT_ID environment variable is not set and gcloud project could not be determined.")

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

