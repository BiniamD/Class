"""
Initialization script for FinancialIQ
Handles system setup and configuration
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv
from logger import FinancialIQLogger
from document_processor import EnhancedSECFilingProcessor
from cache_manager import CacheManager
from sec_filing_rag_system import FinancialIQSystem

class FinancialIQInitializer:
    """Handles initialization of the FinancialIQ system"""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the initializer"""
        self.logger = FinancialIQLogger(log_dir)
        self.cache_manager = CacheManager()
        self.document_processor = EnhancedSECFilingProcessor()
        self.financial_iq = None
        
    def setup_environment(self) -> None:
        """Setup the environment variables and directories"""
        try:
            # Load environment variables
            load_dotenv()
            
            # Create necessary directories
            os.makedirs("logs", exist_ok=True)
            os.makedirs("cache", exist_ok=True)
            os.makedirs("documents/pdfs", exist_ok=True)
            
            self.logger.info("Environment setup completed")
            
        except Exception as e:
            self.logger.log_error(e, "environment_setup")
            raise

    def initialize_system(
        self,
        project_id: str,
        location: str,
        bucket_name: Optional[str] = None,
        pdf_folder: Optional[str] = None,
        local_dir: Optional[str] = None
    ) -> None:
        """Initialize the FinancialIQ system"""
        try:
            self.logger.info("Starting system initialization...")
            
            # Initialize RAG system
            self.financial_iq = FinancialIQSystem(
                project_id=project_id,
                location=location,
                bucket_name=bucket_name or os.getenv("BUCKET_NAME", "adta5770-docs"),
                pdf_folder=pdf_folder or os.getenv("PDF_FOLDER", "documents/sec_filings_pdf")
            )
            
            if local_dir:
                # Process local documents
                self.process_local_documents(local_dir)
            else:
                # Setup cloud system
                self.financial_iq.setup_system(load_existing=True)
            
            self.logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.logger.log_error(e, "system_initialization")
            raise

    def process_local_documents(self, directory: str) -> None:
        """Process documents from local directory"""
        try:
            self.logger.info(f"Processing documents from {directory}")
            
            # Get PDF files
            pdf_files = [
                os.path.join(directory, f) 
                for f in os.listdir(directory) 
                if f.lower().endswith('.pdf')
            ]
            
            if not pdf_files:
                self.logger.warning(f"No PDF files found in {directory}")
                return
            
            # Process each document
            for pdf_file in pdf_files:
                # Check cache first
                cached_result = self.cache_manager.get_document_cache(pdf_file)
                
                if cached_result:
                    self.logger.info(f"Using cached result for {pdf_file}")
                    result = cached_result
                else:
                    # Process document
                    result = self.document_processor.process_document(pdf_file)
                    self.cache_manager.set_document_cache(pdf_file, result)
                    self.logger.log_document_processing(pdf_file, result)
                
                # Add to vector store
                self.financial_iq.add_to_vector_store(result)
            
            self.logger.info(f"Processed {len(pdf_files)} documents")
            
        except Exception as e:
            self.logger.log_error(e, "document_processing")
            raise

    def verify_setup(self) -> bool:
        """Verify that the system is properly initialized"""
        try:
            if not self.financial_iq:
                self.logger.error("FinancialIQ system not initialized")
                return False
            
            # Verify vector store
            if not self.financial_iq.vector_store:
                self.logger.error("Vector store not initialized")
                return False
            
            # Verify LLM
            if not self.financial_iq.llm:
                self.logger.error("LLM not initialized")
                return False
            
            self.logger.info("System verification completed successfully")
            return True
            
        except Exception as e:
            self.logger.log_error(e, "system_verification")
            return False

def main():
    """Main entry point for initialization"""
    try:
        # Create initializer
        initializer = FinancialIQInitializer()
        
        # Setup environment
        initializer.setup_environment()
        
        # Get configuration from environment or command line
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if len(sys.argv) > 1:
            # Use command line arguments if provided
            project_id = sys.argv[1]
            if len(sys.argv) > 2:
                location = sys.argv[2]
        
        if not project_id:
            raise ValueError("Project ID must be provided either through environment variable or command line")
        
        # Initialize system
        initializer.initialize_system(
            project_id=project_id,
            location=location
        )
        
        # Verify setup
        if initializer.verify_setup():
            print("FinancialIQ system initialized successfully!")
        else:
            print("System initialization failed verification")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 