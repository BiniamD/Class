# Directory structure
preprocessing/
├── __init__.py
├── data_loader.py
├── feature_engineer.py
├── data_validator.py
├── sequence_creator.py
├── utils.py
└── main.py

# data_loader.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Union

class DataLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and perform initial cleaning of data"""
        try:
            # Read the data
            df = pd.read_csv(file_path)
            
            # Convert date to datetime
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # Sort by date and symbol
            df = df.sort_values(['Symbol', 'Date'])
            
            self.logger.info(f"Loaded data shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame, train_end: str = '2023-12-31',
                   val_end: str = '2024-06-30') -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        try:
            train = df[df['Date'] <= train_end]
            val = df[(df['Date'] > train_end) & (df['Date'] <= val_end)]
            test = df[df['Date'] > val_end]
            
            splits = {
                'train': train,
                'val': val,
                'test': test
            }
            
            for split_name, split_data in splits.items():
                self.logger.info(f"{split_name} set shape: {split_data.shape}")
                
            return splits
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

# feature_engineer.py
import pandas as pd
import numpy as np
from typing import List
import logging
import ta

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators and features"""
        try:
            df = df.copy()
            
            # Additional Moving Averages
            df['MA50'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=50).mean())
            df['MA200'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.rolling(window=200).mean())
            
            # Bollinger Band Width
            df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
            
            # ATR calculation
            df['TR'] = df.groupby('Symbol').apply(
                lambda x: ta.volatility.true_range(x['High'], x['Low'], 
                x['Close'])).reset_index(level=0, drop=True)
            df['ATR'] = df.groupby('Symbol')['TR'].transform(
                lambda x: x.rolling(window=14).mean())
            
            # ROC (Rate of Change)
            df['ROC'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x.pct_change(periods=10) * 100)
            
            # Additional Momentum Indicators
            df['Momentum'] = df.groupby('Symbol')['Close'].transform(
                lambda x: x - x.shift(10))
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            raise
            
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all features appropriately"""
        try:
            df = df.copy()
            
            # Price normalization
            price_cols = ['Open', 'High', 'Low', 'Close', 'MA20', 'MA50', 'MA200']
            for col in price_cols:
                df[col] = df.groupby('Symbol')[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min()))
            
            # Volume normalization (log transform)
            df['Volume'] = np.log1p(df['Volume'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            raise

# sequence_creator.py
class SequenceCreator:
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.logger = logging.getLogger(__name__)
        
    def create_sequences(self, df: pd.DataFrame, 
                        feature_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for deep learning model"""
        try:
            sequences = []
            targets = []
            
            # Create sequences for each symbol
            for symbol in df['Symbol'].unique():
                symbol_data = df[df['Symbol'] == symbol]
                
                for i in range(len(symbol_data) - self.sequence_length):
                    # Extract sequence
                    sequence = symbol_data[feature_columns].iloc[i:(i + self.sequence_length)].values
                    target = symbol_data['Returns'].iloc[i + self.sequence_length]
                    
                    sequences.append(sequence)
                    targets.append(target)
            
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error creating sequences: {str(e)}")
            raise

# main.py
import logging
from pathlib import Path
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from sequence_creator import SequenceCreator
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize components
        data_loader = DataLoader()
        feature_engineer = FeatureEngineer()
        sequence_creator = SequenceCreator(sequence_length=20)
        
        # Load data
        logger.info("Loading data...")
        df = data_loader.load_data('sp500_master_data.csv')
        
        # Calculate features
        logger.info("Calculating features...")
        df = feature_engineer.calculate_features(df)
        
        # Split data
        logger.info("Splitting data...")
        splits = data_loader.split_data(df)
        
        # Create sequences for each split
        sequences = {}
        for split_name, split_data in splits.items():
            logger.info(f"Creating sequences for {split_name} split...")
            
            # Normalize features
            split_data = feature_engineer.normalize_features(split_data)
            
            # Create sequences
            X, y = sequence_creator.create_sequences(
                split_data,
                feature_columns=['Close', 'Returns', 'RSI', 'MACD', 'MA20', 
                               'BB_Width', 'ATR', 'ROC', 'Volume']
            )
            
            sequences[split_name] = {
                'X': X,
                'y': y
            }
            
            logger.info(f"{split_name} sequences shape: X={X.shape}, y={y.shape}")
        
        # Save preprocessed data
        logger.info("Saving preprocessed data...")
        output_dir = Path('preprocessed_data')
        output_dir.mkdir(exist_ok=True)
        
        for split_name, split_sequences in sequences.items():
            np.save(output_dir / f'X_{split_name}.npy', split_sequences['X'])
            np.save(output_dir / f'y_{split_name}.npy', split_sequences['y'])
        
        logger.info("Preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main preprocessing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
