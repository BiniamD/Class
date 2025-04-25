import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SP500DataCollector:
    def __init__(self):
        self.sp500_tickers = self._get_sp500_tickers()
        
    def _get_sp500_tickers(self) -> list:
        """Get S&P 500 tickers using Wikipedia"""
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        return df['Symbol'].tolist()
    
    def fetch_and_store_data(self, start_date: str, end_date: str, output_dir: str = 'stock_data'):
        """Fetch data for all S&P 500 companies and store in CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Create a master DataFrame to store all data
        master_df = pd.DataFrame()
        
        for ticker in self.sp500_tickers:
            try:
                # Fetch data
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                # Add ticker column
                df['Symbol'] = ticker
                
                # Calculate technical indicators
                df = self._calculate_technical_indicators(df)
                
                # Add to master DataFrame
                master_df = pd.concat([master_df, df])
                
                print(f"Successfully processed {ticker}")
                
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Save master DataFrame
        master_file = os.path.join(output_dir, 'sp500_master_data.csv')
        master_df.to_csv(master_file)
        
        # Save summary statistics
        summary = {
            'total_companies': len(self.sp500_tickers),
            'total_observations': len(master_df),
            'date_range': f"{start_date} to {end_date}",
            'file_location': master_file
        }
        
        with open(os.path.join(output_dir, 'data_summary.txt'), 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
                
        return master_file, summary
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['20dSTD'] = df['Close'].rolling(window=20).std()
        df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
        df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
        
        # Add percentage change
        df['Returns'] = df['Close'].pct_change()
        
        # Add volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        return df

def main():
    # Initialize collector
    collector = SP500DataCollector()
    
    # Set date range for 5 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)  # 5 years
    
    # Fetch and store data
    master_file, summary = collector.fetch_and_store_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    print("\nData Collection Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()