import pandas as pd
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_analyze_data():
    """
    Load and analyze the SP500 master data
    """
    try:
        # Load the data
        logger.info("Loading SP500 master data...")
        df = pd.read_csv('sp500_master_data.csv')
        
        # Basic data analysis
        logger.info("\nDataset Overview:")
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Number of unique stocks: {df['ticker'].nunique()}")
        
        # Check missing values
        missing_values = df.isnull().sum() / len(df) * 100
        logger.info("\nFeatures with missing values (%):")
        logger.info(missing_values[missing_values > 0])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_stock_data(df: pd.DataFrame, ticker: str = None):
    """
    Prepare data for a single stock or all stocks
    """
    try:
        # Filter for specific stock if provided
        if ticker:
            df = df[df['ticker'] == ticker].copy()
            logger.info(f"\nPreparing data for {ticker}")
        
        # Sort by date
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Initialize preprocessor
        from financial_preprocessor import EnhancedFinancialPreprocessor
        preprocessor = EnhancedFinancialPreprocessor(sequence_length=20)
        
        # Prepare data
        prepared_data = preprocessor.prepare_data(df)
        
        # Log preparation summary
        logger.info("\nData Preparation Summary:")
        logger.info(f"Training sequences: {prepared_data['train']['X'].shape}")
        logger.info(f"Validation sequences: {prepared_data['val']['X'].shape}")
        
        return prepared_data
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def visualize_features(df: pd.DataFrame, ticker: str):
    """
    Create visualization of key features for a stock
    """
    try:
        # Filter data for the specified stock
        stock_data = df[df['ticker'] == ticker].copy()
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        
        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Price and Moving Averages
        axes[0].plot(stock_data['Date'], stock_data['Close'], label='Close')
        axes[0].plot(stock_data['Date'], stock_data['MA_20'], label='MA_20')
        axes[0].plot(stock_data['Date'], stock_data['MA_50'], label='MA_50')
        axes[0].set_title(f'{ticker} - Price and Moving Averages')
        axes[0].legend()
        
        # Technical Indicators
        axes[1].plot(stock_data['Date'], stock_data['RSI_14'], label='RSI_14')
        axes[1].plot(stock_data['Date'], stock_data['MACD'], label='MACD')
        axes[1].set_title('Technical Indicators')
        axes[1].legend()
        
        # Volume
        axes[2].bar(stock_data['Date'], stock_data['Volume'])
        axes[2].set_title('Volume')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    try:
        # Load and analyze data
        df = load_and_analyze_data()
        
        # Example: Prepare data for a single stock (e.g., AAPL)
        aapl_data = prepare_stock_data(df, 'AAPL')
        
        # Visualize features
        visualize_features(df, 'AAPL')
        
        # Optional: Prepare data for all stocks
        # all_stocks_data = prepare_stock_data(df)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
