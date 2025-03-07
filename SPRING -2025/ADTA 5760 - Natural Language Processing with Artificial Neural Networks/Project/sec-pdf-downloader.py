import pandas as pd
import requests
import os
from typing import Optional
import time
from urllib.parse import quote

class SECPDFDownloader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sec-api.io/filing-reader"
        self.failed_downloads = []

    def download_pdf(self, filing_url: str, output_dir: str, filename: Optional[str] = None) -> bool:
        """
        Download a filing as PDF using the SEC API.
        
        Args:
            filing_url (str): URL of the filing
            output_dir (str): Directory to save the PDF
            filename (str, optional): Custom filename for the PDF
            
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            # Construct the API URL
            params = {
                'token': self.api_key,
                'url': filing_url
            }
            
            print(f"Downloading: {filing_url}")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code == 429:  # Rate limit
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.download_pdf(filing_url, output_dir, filename)
            
            response.raise_for_status()
            
            # Generate filename if not provided
            if not filename:
                parts = filing_url.split('/')
                filename = f"filing_{parts[-1].replace('.htm', '')}.pdf"
            
            # Ensure .pdf extension
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the PDF
            output_path = os.path.join(output_dir, filename)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Successfully downloaded: {output_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {filing_url}: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            self.failed_downloads.append((filing_url, str(e)))
            return False
        except Exception as e:
            print(f"Unexpected error downloading {filing_url}: {e}")
            self.failed_downloads.append((filing_url, str(e)))
            return False

def main():
    # Your API key
    API_KEY = "b3eacaa72e15fec1ef69bbd2a71dcbb790fdf6ab38c852436df88f306b98d2d0"
    
    # Create downloader instance
    downloader = SECPDFDownloader(API_KEY)
    
    # Output directory for PDFs
    output_dir = "sec_filings_pdf"
    
    try:
        # Read the CSV file
        df = pd.read_csv('sec_filings.csv')
        
        print(f"Found {len(df)} filings in CSV file")
        
        # Create counters
        successful = 0
        failed = 0
        
        # Download each filing
        for index, row in df.iterrows():
            try:
                filing_url = row['filing_url']
                if pd.isna(filing_url) or not filing_url:
                    print(f"Skipping row {index}: No filing URL")
                    continue
                
                # Create filename using company name and form type
                safe_company_name = "".join(x for x in row['companyName'] if x.isalnum() or x in [' ', '-', '_'])
                filename = f"{safe_company_name}_{row['formType']}_{row['filedAt'][:10]}.pdf"
                
                # Download the PDF
                if downloader.download_pdf(filing_url, output_dir, filename):
                    successful += 1
                else:
                    failed += 1
                
                # Add a small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                failed += 1
                continue
        
        print(f"\nDownload summary:")
        print(f"Successfully downloaded: {successful} / {len(df)} filings")
        print(f"Failed downloads: {failed}")
        print(f"PDFs saved in: {os.path.abspath(output_dir)}")
        
        if downloader.failed_downloads:
            print("\nFailed downloads details:")
            for url, error in downloader.failed_downloads:
                print(f"URL: {url}")
                print(f"Error: {error}\n")
        
    except FileNotFoundError:
        print("Error: sec_filings.csv not found in the current directory!")
        print(f"Current directory: {os.getcwd()}")
        print("Please make sure the CSV file exists in this location.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
