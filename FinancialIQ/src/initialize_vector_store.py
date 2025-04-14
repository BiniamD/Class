"""
Initialize Vector Store
This script prepares the vector store independently of the main application.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv
import json

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Load environment variables
load_dotenv()

from src.rag_system import RAGSystem, VectorStoreConfig
from src.document_processor import EnhancedSECFilingProcessor
from src.logger import FinancialIQLogger

def setup_gcp_credentials():
    """
    Set up GCP credentials using Application Default Credentials (ADC).
    Supports both service account and user credentials.
    """
    try:
        use_service_account = os.getenv("USE_SERVICE_ACCOUNT", "false").lower() == "true"
        
        if use_service_account:
            service_account_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH")
            if not service_account_path:
                raise ValueError("SERVICE_ACCOUNT_KEY_PATH must be set when using service account")
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
            logging.info("Using service account credentials")
        else:
            logging.info("Using default application credentials")
            
    except Exception as e:
        logging.error(f"Error setting up GCP credentials: {str(e)}")
        raise

def initialize_vector_store(
    config: Optional[VectorStoreConfig] = None
) -> bool:
    """
    Initialize the vector store with documents from either GCS or local directory.
    Uses environment variables for configuration.
    
    Args:
        config: Vector store configuration (optional)
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    # Initialize logger first
    logger = FinancialIQLogger("logs")
    
    try:
        # Set up GCP credentials
        setup_gcp_credentials()
        
        # Get configuration from environment variables
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")  # Changed from PROJECT_ID
        location = os.getenv("GOOGLE_CLOUD_LOCATION")   # Changed from LOCATION
        bucket_name = os.getenv("GCS_BUCKET_NAME")      # Changed from BUCKET_NAME
        pdf_folder = os.getenv("PDF_FOLDER")
        local_dir = os.getenv("LOCAL_DIR")
        
        if not project_id or not location:
            raise ValueError("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION must be set in .env file")
        
        logger.info("Starting vector store initialization")
        
        # Initialize document processor
        processor = EnhancedSECFilingProcessor(
            project_id=project_id,
            bucket_name=bucket_name
        )
        
        # Initialize RAG system
        rag_system = RAGSystem(
            model_name="all-MiniLM-L6-v2",
            config=config or VectorStoreConfig(
                save_path="data/vector_store",
                chunk_size=1000,
                chunk_overlap=200,
                create_backup=True,
                validate_store=True
            )
        )
        
        # Process documents
        processed_docs = []
        
        if bucket_name and pdf_folder:
            logger.info(f"Processing documents from GCS: {bucket_name}/{pdf_folder}")
            processed_docs = processor.process_gcs_directory(pdf_folder)
        elif local_dir:
            logger.info(f"Processing documents from local directory: {local_dir}")
            for file in Path(local_dir).glob("*.pdf"):
                result = processor.process_filing(str(file))
                if result:
                    processed_docs.append(result)
        else:
            logger.error("No document source specified in .env file")
            return False
        
        if not processed_docs:
            logger.error("No documents were processed successfully")
            return False
        
        # Create vector store
        logger.info(f"Creating vector store with {len(processed_docs)} documents")
        vector_store = rag_system.create_vector_store(processed_docs)
        
        if not vector_store:
            logger.error("Failed to create vector store")
            return False
        
        # Save processed files list
        processed_files = {
            str(doc.get("source", "unknown")): datetime.now().isoformat()
            for doc in processed_docs
        }
        
        with open("processed_files.json", "w") as f:
            json.dump(processed_files, f)
        
        logger.info("Vector store initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during vector store initialization: {str(e)}")
        return False

if __name__ == "__main__":
    success = initialize_vector_store()
    
    if success:
        print("Vector store initialization completed successfully")
        sys.exit(0)
    else:
        print("Vector store initialization failed")
        sys.exit(1) 