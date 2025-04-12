"""
FinancialIQ: SEC Filing Metadata Handler
Handles processing and management of SEC filing metadata from CSV files
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import csv

class SECFilingMetadata:
    """Handles SEC filing metadata from CSV files"""
    
    def __init__(self, file_path: str):
        """Initialize with path to CSV file"""
        self.file_path = file_path
        self.metadata = {
            "file_name": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
            "processing_date": datetime.now().isoformat()
        }
        self.metadata_df = None
        self.load_metadata()
    
    def load_metadata(self) -> None:
        """Load and preprocess metadata from CSV"""
        self.metadata_df = pd.read_csv(self.file_path)
        
        # Convert filing date to datetime
        self.metadata_df['filedAt'] = pd.to_datetime(self.metadata_df['filedAt'])
        
        # Clean company names
        self.metadata_df['companyName'] = self.metadata_df['companyName'].str.strip()
        
        # Create a clean form type column
        self.metadata_df['cleanFormType'] = self.metadata_df['formType'].str.replace('/A', '').str.strip()
    
    def get_companies(self) -> List[str]:
        """Get unique company names"""
        return sorted(self.metadata_df['companyName'].unique().tolist())
    
    def get_form_types(self) -> List[str]:
        """Get unique form types"""
        return sorted(self.metadata_df['cleanFormType'].unique().tolist())
    
    def get_filings_by_company(self, company_name: str) -> pd.DataFrame:
        """Get all filings for a specific company"""
        return self.metadata_df[self.metadata_df['companyName'] == company_name]
    
    def get_filings_by_form_type(self, form_type: str) -> pd.DataFrame:
        """Get all filings of a specific form type"""
        return self.metadata_df[self.metadata_df['cleanFormType'] == form_type]
    
    def get_recent_filings(self, days: int = 30) -> pd.DataFrame:
        """Get filings from the last N days"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        return self.metadata_df[self.metadata_df['filedAt'] >= cutoff_date]
    
    def get_filing_metadata(self, accession_no: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific filing by accession number"""
        filing = self.metadata_df[self.metadata_df['accessionNo'] == accession_no]
        if len(filing) == 0:
            return None
        
        return filing.iloc[0].to_dict()
    
    def get_filing_url(self, accession_no: str) -> Optional[str]:
        """Get the filing URL for a specific accession number"""
        filing = self.metadata_df[self.metadata_df['accessionNo'] == accession_no]
        if len(filing) == 0:
            return None
        
        return filing.iloc[0]['filing_url']
    
    def get_filing_statistics(self) -> Dict[str, Any]:
        """Get summary statistics about the filings"""
        return {
            'total_filings': len(self.metadata_df),
            'unique_companies': len(self.metadata_df['companyName'].unique()),
            'form_type_counts': self.metadata_df['cleanFormType'].value_counts().to_dict(),
            'latest_filing_date': self.metadata_df['filedAt'].max(),
            'earliest_filing_date': self.metadata_df['filedAt'].min()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return self.metadata
    
    def save_to_csv(self, csv_path: str):
        """Save metadata to CSV file"""
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metadata.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(self.metadata)
    
    @classmethod
    def load_from_csv(cls, csv_path: str) -> Dict[str, Any]:
        """Load metadata from CSV file"""
        if not os.path.exists(csv_path):
            return {}
        
        metadata = {}
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = row['file_name']
                metadata[file_name] = row
        
        return metadata 