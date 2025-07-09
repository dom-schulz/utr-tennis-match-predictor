from google.cloud import compute_v1
import functions_framework
import json

# Triggered from message on Cloud Pub/Sub topic.
@functions_framework.cloud_event
def start_vm(cloud_event):
    # Project ID and zone
    project_id = "cpsc324-project-452600"
    zone = "us-west1-a"
    instance_name = "matches-scraper-vm"
    
    # Log the incoming event for debugging
    print(f"Received Pub/Sub event: {cloud_event}")
    
    try:
        # Create the instance client
        instance_client = compute_v1.InstancesClient()
        
        # Start the instance
        operation = instance_client.start(
            project=project_id,
            zone=zone,
            instance=instance_name
        )
        
        print(f"VM start operation initiated: {operation.name}")
        return f"VM start operation initiated: {operation.name}"
    except Exception as e:
        print(f"Error starting VM: {str(e)}")
        raise e