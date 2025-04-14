"""
Script to set up required GCS folder structure
"""

import os
from google.cloud import storage
from dotenv import load_dotenv

def setup_gcs_folders():
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    pdf_folder = os.getenv("PDF_FOLDER")
    
    print(f"Setting up GCS folders for project: {project_id}")
    print(f"Bucket: {bucket_name}")
    
    try:
        # Initialize the client
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        if not bucket.exists():
            print(f"\n❌ Bucket '{bucket_name}' does not exist!")
            return
        
        print(f"\n✅ Bucket '{bucket_name}' exists")
        
        # Create required folders
        folders = [
            pdf_folder,
            "processed/",
            "cache/metadata/",
            "cache/vector_store/"
        ]
        
        for folder in folders:
            # Create a dummy file to establish the folder
            blob = bucket.blob(f"{folder}.placeholder")
            blob.upload_from_string("")
            print(f"✅ Created folder: {folder}")
        
        print("\nGCS folder structure setup completed!")
        
    except Exception as e:
        print(f"\n❌ Error setting up GCS folders: {str(e)}")

if __name__ == "__main__":
    setup_gcs_folders() 