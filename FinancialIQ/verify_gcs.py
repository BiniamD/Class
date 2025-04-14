"""
Script to verify GCS bucket locations and contents
"""

import os
from google.cloud import storage
from dotenv import load_dotenv

def verify_gcs_locations():
    # Load environment variables
    load_dotenv()
    
    # Get configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    bucket_name = os.getenv("GCS_BUCKET_NAME")
    pdf_folder = os.getenv("PDF_FOLDER")
    
    print(f"Verifying GCS locations for project: {project_id}")
    print(f"Bucket: {bucket_name}")
    print(f"PDF Folder: {pdf_folder}")
    print("\nChecking locations...")
    
    try:
        # Initialize the client
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Check if bucket exists
        if not bucket.exists():
            print(f"\n❌ Bucket '{bucket_name}' does not exist!")
            return
        
        print(f"\n✅ Bucket '{bucket_name}' exists")
        
        # Check PDF folder
        pdf_blobs = list(bucket.list_blobs(prefix=pdf_folder))
        if pdf_blobs:
            print(f"\n✅ PDF folder '{pdf_folder}' exists with {len(pdf_blobs)} files")
            print("\nPDF files found:")
            for blob in pdf_blobs:
                print(f"- {blob.name}")
        else:
            print(f"\n⚠️ PDF folder '{pdf_folder}' exists but is empty")
        
        # Check processed folder
        processed_blobs = list(bucket.list_blobs(prefix="processed/"))
        if processed_blobs:
            print(f"\n✅ Processed folder exists with {len(processed_blobs)} files")
            print("\nProcessed files found:")
            for blob in processed_blobs:
                print(f"- {blob.name}")
        else:
            print("\n⚠️ Processed folder exists but is empty")
        
        # Check cache folder
        cache_blobs = list(bucket.list_blobs(prefix="cache/"))
        if cache_blobs:
            print(f"\n✅ Cache folder exists with {len(cache_blobs)} files")
        else:
            print("\n⚠️ Cache folder exists but is empty")
            
    except Exception as e:
        print(f"\n❌ Error verifying GCS locations: {str(e)}")

if __name__ == "__main__":
    verify_gcs_locations() 