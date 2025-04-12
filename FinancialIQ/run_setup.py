import os
from dotenv import load_dotenv
from src.sec_filing_rag_system import FinancialIQSystem

def main():
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment variables
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
    bucket_name = os.getenv('GCS_BUCKET_NAME')
    pdf_folder = os.getenv('PDF_FOLDER', 'sec_filings')
    metadata_csv_path = os.getenv('METADATA_CSV_PATH', 'documents/documents_sec_filings.csv')
    
    # Initialize the system
    system = FinancialIQSystem(
        project_id=project_id,
        location=location,
        bucket_name=bucket_name,
        pdf_folder=pdf_folder,
        metadata_csv_path=metadata_csv_path
    )
    
    # Setup the system
    print("Setting up the FinancialIQ system...")
    system.setup_system(load_existing=True)
    
    # Test the system with a sample query
    test_query = "What are the most recent 10-K filings available?"
    print("\nTesting the system with a sample query:", test_query)
    result = system.process_query(test_query)
    
    print("\nAnswer:", result['answer'])
    print("\nSources:")
    for source in result['sources']:
        print(f"- {source}")

if __name__ == "__main__":
    main() 