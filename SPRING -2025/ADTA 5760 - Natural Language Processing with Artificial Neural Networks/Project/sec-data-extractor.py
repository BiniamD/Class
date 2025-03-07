import requests
import pandas as pd
from typing import List, Dict, Any
import time

class SECDataExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.sec-api.io"

    def get_filings(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch SEC filings using the EDGAR API endpoint.
        
        Args:
            limit (int): Number of filings to retrieve (default: 100)
            
        Returns:
            List[Dict]: List of filing data
        """
        try:
            # Construct the query according to documentation
            payload = {
                "query": f"formType:\"10-K\" OR formType:\"10-Q\" OR formType:\"8-K\"",
                "from": 0,
                "size": limit,
                "sort": [{"filedAt": {"order": "desc"}}]
            }

            # Make request to the EDGAR endpoint with token as a parameter
            url = f"{self.base_url}/query"
            params = {
                'token': self.api_key
            }

            print(f"Making request to: {url}")  # Debug print
            response = requests.get(url, params=params, json=payload)

            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                print(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                return self.get_filings(limit)

            response.raise_for_status()
            
            data = response.json()
            
            # Process the filings
            processed_filings = []
            for filing in data:
                processed_filings.append({
                    'accessionNo': filing.get('accessionNo', ''),
                    'cik': filing.get('cik', ''),
                    'ticker': filing.get('ticker', ''),
                    'companyName': filing.get('companyName', ''),
                    'formType': filing.get('formType', ''),
                    'filedAt': filing.get('filedAt', ''),
                    'filing_url': filing.get('linkToFilingDetails', '')
                })
            
            return processed_filings

        except requests.exceptions.RequestException as e:
            print(f"Error fetching SEC filings: {e}")
            if hasattr(e, 'response'):
                if e.response.status_code == 401:
                    print("Invalid API key or unauthorized access")
                elif e.response.status_code == 403:
                    print("Access forbidden - check your subscription status")
                if hasattr(e.response, 'text'):
                    print(f"Response text: {e.response.text}")
            return []

    def save_to_csv(self, filings: List[Dict[str, Any]], filename: str = 'sec_filings.csv'):
        """
        Save the filing data to a CSV file.
        
        Args:
            filings (List[Dict]): List of filing data
            filename (str): Output filename
        """
        if not filings:
            print("No data to save")
            return
            
        df = pd.DataFrame(filings)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

def main():
    # Your API key
    API_KEY = "b3eacaa72e15fec1ef69bbd2a71dcbb790fdf6ab38c852436df88f306b98d2d0"
    
    # Initialize the extractor
    extractor = SECDataExtractor(API_KEY)
    
    # Get the filings
    print("Fetching SEC filings...")
    filings = extractor.get_filings(limit=100)
    
    if filings:
        print(f"\nRetrieved {len(filings)} filings")
        
        # Display first few entries
        print("\nSample of retrieved data:")
        for filing in filings[:5]:
            print(f"\nAccession No: {filing['accessionNo']}")
            print(f"CIK: {filing['cik']}")
            print(f"Ticker: {filing['ticker']}")
            print(f"Company: {filing['companyName']}")
            print(f"Form Type: {filing['formType']}")
            print(f"Filed At: {filing['filedAt']}")
            print(f"Filing URL: {filing['filing_url']}")
        
        # Save to CSV
        extractor.save_to_csv(filings)
    else:
        print("\nTroubleshooting tips:")
        print("1. Verify your API key is valid")
        print("2. Check if you have hit your API rate limits")
        print("3. Ensure you have an active subscription")

if __name__ == "__main__":
    main()
