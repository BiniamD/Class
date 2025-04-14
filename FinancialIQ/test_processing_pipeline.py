"""
Script to test the complete processing pipeline with detailed error logging
"""

import os
import sys
import json
import traceback
from google.cloud import storage
from dotenv import load_dotenv
from src.document_processor import EnhancedSECFilingProcessor
from src.logger import FinancialIQLogger
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import vertexai
import logging
from google.cloud import aiplatform
from google.api_core import exceptions
from typing import List, Dict, Any
import time
from datetime import datetime
from langchain_community.vectorstores import MatchingEngine
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account # For loading SA key file

def extract_text_from_json(json_obj):
    """Recursively extract text from JSON object"""
    text_parts = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, (dict, list)):
                text_parts.extend(extract_text_from_json(value))
    elif isinstance(json_obj, list):
        for item in json_obj:
            text_parts.extend(extract_text_from_json(item))
    elif isinstance(json_obj, str):
        text_parts.append(json_obj)
    return text_parts

def test_processing_pipeline():
    try:
        # Load environment variables
        load_dotenv()
        
        # Load all required configuration from .env
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        bucket_name = os.getenv('GCS_BUCKET_NAME')
        embeddings_bucket = os.getenv('VERTEX_AI_EMBEDDINGS_BUCKET')
        index_id = os.getenv('VERTEX_AI_INDEX_ID')
        endpoint_id = os.getenv('VERTEX_AI_ENDPOINT_ID')
        embeddings_model = os.getenv('VERTEX_AI_EMBEDDING_MODEL', 'textembedding-gecko@001')
        chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        
        # Validate required environment variables
        required_vars = {
            'GOOGLE_CLOUD_PROJECT': project_id,
            'GCS_BUCKET_NAME': bucket_name,
            'VERTEX_AI_EMBEDDINGS_BUCKET': embeddings_bucket,
            'VERTEX_AI_INDEX_ID': index_id,
            'VERTEX_AI_ENDPOINT_ID': endpoint_id
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        print("\nConfiguration loaded from .env:")
        print(f"  ✅ Project ID: {project_id}")
        print(f"  ✅ Location: {location}")
        print(f"  ✅ GCS Bucket: {bucket_name}")
        print(f"  ✅ Embeddings Bucket: {embeddings_bucket}")
        print(f"  ✅ Index ID: {index_id}")
        print(f"  ✅ Endpoint ID: {endpoint_id}")
        print(f"  ✅ Embedding Model: {embeddings_model}")
        print(f"  ✅ Chunk Size: {chunk_size}")
        print(f"  ✅ Chunk Overlap: {chunk_overlap}")
        
        # Initialize Vertex AI with the correct project
        vertexai.init(project=project_id, location=location)
        
        try:
            # Initialize Vertex AI resources
            index = aiplatform.MatchingEngineIndex(
                index_name=index_id,
                project=project_id,
                location=location
            )
            endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=endpoint_id,
                project=project_id,
                location=location
            )
            
            print(f"\nVertex AI Resources:")
            print(f"  ✅ Using Vertex AI Index: {index.name}")
            print(f"  ✅ Using Vertex AI Endpoint: {endpoint.name}")
            
        except Exception as e:
            print(f"  ❌ Error accessing Vertex AI resources: {str(e)}")
            print("  Please ensure you have run setup_vertex_ai.py first")
            sys.exit(1)

        # Step 1: Verify necessary folders exist
        print("\nStep 1: Verifying necessary folders")
        required_folders = [
            os.getenv('PDF_FOLDER', 'documents/sec_filings_pdf'),
            'processed/',
            os.getenv('CACHE_METADATA_DIR', 'cache/metadata'),
            os.getenv('CACHE_VECTOR_STORE_DIR', 'cache/vector_store')
        ]

        for folder in required_folders:
            try:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                    print(f"  ✅ Created folder: {folder}")
                else:
                    print(f"  ✅ Found folder: {folder}")
            except Exception as e:
                print(f"  ❌ Error creating folder {folder}: {str(e)}")
                sys.exit(1)

        # Initialize GCS client
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
        except Exception as e:
            print(f"  ❌ Error initializing GCS client: {str(e)}")
            sys.exit(1)

        # Step 3: Find processed files and metadata
        print("\nStep 3: Finding processed files and metadata")
        try:
            # Get list of processed files
            processed_blobs = list(bucket.list_blobs(prefix='processed/documents/sec_filings_pdf'))
            print(f"  ✅ Found {len(processed_blobs)} processed files")

            # Get list of metadata files
            metadata_blobs = list(bucket.list_blobs(prefix='cache/metadata/documents/sec_filings_pdf'))
            print(f"  ✅ Found {len(metadata_blobs)} metadata files")

            if not processed_blobs:
                print("  ❌ No processed files found")
                sys.exit(1)

            # Load existing documents for embedding generation
            print("\nStep 4: Loading processed documents")
            all_documents = load_existing_documents(bucket, processed_blobs)
            if not all_documents:
                print("  ❌ No documents available for embedding generation")
                sys.exit(1)
            print(f"  ✅ Loaded {len(all_documents)} documents for embedding generation")

            # Step 5: Generate and Upload Embeddings
            print("\nStep 5: Generating and Uploading Embeddings")
            return generate_and_upload_embeddings(
                all_documents,
                index,
                endpoint,
                embeddings_bucket,
                storage_client,
                project_id,
                location,
                embeddings_model,
                chunk_size,
                chunk_overlap
            )

        except Exception as e:
            print(f"  ❌ Error accessing processed files and metadata: {str(e)}")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Critical error in processing pipeline: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        sys.exit(1)

def load_existing_documents(bucket, processed_blobs):
    """Load existing processed documents from GCS"""
    all_documents = []
    has_errors = False
    
    for blob in processed_blobs[0:1]: # Only process the first file for testing
        try:
            # Load the processed JSON content
            processed_content = json.loads(blob.download_as_text())
            
            # Try different possible structures to extract text
            text_content = None
            if isinstance(processed_content, dict):
                # Try direct text field
                text_content = processed_content.get('text')
                
                # Try nested text field
                if not text_content and 'data' in processed_content:
                    text_content = processed_content['data'].get('text')
                
                # Try content field
                if not text_content:
                    text_content = processed_content.get('content')
                
                # Try extracting from sections
                if not text_content and 'sections' in processed_content:
                    sections = processed_content.get('sections', [])
                    text_parts = []
                    for section in sections:
                        if isinstance(section, dict):
                            text_parts.append(section.get('text', ''))
                        elif isinstance(section, str):
                            text_parts.append(section)
                    text_content = '\n\n'.join(filter(None, text_parts))
            
            # If still no text content, try to extract from any string values
            if not text_content:
                text_parts = []
                def extract_text(obj):
                    if isinstance(obj, str):
                        text_parts.append(obj)
                    elif isinstance(obj, dict):
                        for value in obj.values():
                            extract_text(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            extract_text(item)
                
                extract_text(processed_content)
                text_content = '\n\n'.join(filter(None, text_parts))
            
            if not text_content:
                print(f"  ⚠️ Warning: No text content found in {blob.name}")
                has_errors = True
                continue

            # Extract metadata
            metadata_content = {}
            if isinstance(processed_content, dict):
                metadata_content = processed_content.get('metadata', {})
                if not metadata_content and 'data' in processed_content:
                    metadata_content = processed_content['data'].get('metadata', {})

            # Create Document object with the extracted content
            doc = Document(
                page_content=text_content,
                metadata={
                    'source': blob.name,
                    'processed_json': blob.name,
                    'metadata': metadata_content,
                    'document_type': 'sec_filing',
                    'filing_type': metadata_content.get('form_type', ''),
                    'company_name': metadata_content.get('company_name', '')
                }
            )
            all_documents.append(doc)
            print(f"  ✅ Loaded document: {blob.name}")
        except Exception as e:
            print(f"  ⚠️ Error loading processed document {blob.name}: {str(e)}")
            has_errors = True
            continue

    if has_errors:
        print("\n⚠️ Some errors occurred while loading existing documents")
    return all_documents

def generate_and_upload_embeddings(all_documents, index, endpoint, embeddings_bucket, storage_client, project_id, location, embeddings_model, chunk_size, chunk_overlap):
    """Generate and upload embeddings for documents"""
    if not all_documents:
        print("  ❌ No documents to process")
        sys.exit(1)
    
    # embedded_docs = None # Removed test variable
    try:
        # === Logging for Debugging Credentials ===
        print("\n--- Credential & Environment Diagnostics ---")
        key_file_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        print(f"  Env Var GOOGLE_APPLICATION_CREDENTIALS: {key_file_path if key_file_path else 'Not Set'}")
        # Check relevant environment variables
        env_vars_to_check = [
            'GOOGLE_CLOUD_PROJECT', 
            'GOOGLE_CLOUD_QUOTA_PROJECT', 
            'CLOUDSDK_CORE_PROJECT'
        ]
        for var in env_vars_to_check:
            value = os.getenv(var)
            print(f"  Env Var {var}: {value if value else 'Not Set'}")

        # Inspect default credentials found by google-auth
        try:
            credentials, found_project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            # Note: found_project_id might be None if not explicitly in ADC file
            cred_project_id_attr = getattr(credentials, 'project_id', 'N/A (Attribute missing)')
            print(f"  google.auth.default() found credentials for project: {found_project_id if found_project_id else 'Not specified in ADC'}")
            print(f"  Credentials object project_id attribute: {cred_project_id_attr}")
            print(f"  Credentials class: {credentials.__class__.__name__}")
        except Exception as e:
            print(f"  Error inspecting default credentials: {e}")
            credentials = None # Ensure credentials variable exists

        # *** Explicitly load credentials from SA file and set quota project ***
        credentials = None
        if key_file_path:
            try:
                print(f"  Attempting to load credentials from: {key_file_path}")
                credentials = service_account.Credentials.from_service_account_file(
                    key_file_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    quota_project_id=project_id # Explicitly set quota project
                )
                print(f"  Successfully loaded SA credentials. Quota Project enforced: {credentials.quota_project_id}")
                print(f"  Service Account Email: {credentials.service_account_email}")
            except Exception as e:
                print(f"  Error loading SA credentials from file: {e}")
                sys.exit(1)
        else:
            print("  GOOGLE_APPLICATION_CREDENTIALS not set. Cannot proceed with explicit SA loading.")
            # Fallback to default (though we know this has issues)
            # credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            sys.exit(1)

        # Initialize Vertex AI with explicitly loaded credentials
        print(f"  Initializing vertexai with project='{project_id}', location='{location}' using loaded SA credentials")
        vertexai.init(project=project_id, location=location, credentials=credentials)
        print(f"  Vertex AI client initialized. Project: {aiplatform.initializer.global_config.project}")
        print("--- End Diagnostics ---\n")
        # === End Logging ===

        # Create embeddings model with explicitly loaded credentials
        embeddings = VertexAIEmbeddings(
            model_name=embeddings_model, 
            project=project_id, 
            location=location,
            credentials=credentials # Pass explicit credentials
        )
        
        print(f"  ✅ Initialized embedding model: {embeddings_model}")
        print(f"  ✅ Using project: {project_id}")
        print(f"  ✅ Using location: {location}")
        
        # *** Vector Store operations ***
        print("\n  💾 Setting up Vector Store...")
        vector_store = MatchingEngine.from_components(
            project_id=project_id,
            region=location, 
            index_id=index.name,
            endpoint_id=endpoint.name,
            embedding=embeddings,
            gcs_bucket_name=embeddings_bucket
        )
        
        print(f"  ✅ Created MatchingEngine vector store")
        
        # Add documents (This is where the error occurs)
        print(f"  📝 Attempting to add {len(all_documents)} documents to vector store (calls embedding)...")
        vector_store.add_documents(all_documents)
        print(f"  ✅ Successfully added documents to vector store")
        
    except Exception as e:
        print(f"\n--- ERROR OCCURRED --- ")
        print(f"  ❌ Error during embedding/vector store operations: {str(e)}")             
        print(f"  Attempted Project ID: {project_id}")
        print(f"  Attempted Location: {location}")
        print(f"  Attempted Model: {embeddings_model}")
        print("--- End Error --- ")
        sys.exit(1)

    print("\nProcessing pipeline test completed (Vector Store)")
    print("✅ All operations completed successfully")
    sys.exit(0)

if __name__ == "__main__":
    test_processing_pipeline()