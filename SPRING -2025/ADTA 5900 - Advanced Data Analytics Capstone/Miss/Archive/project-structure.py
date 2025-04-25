# %%
# Directory structure
#preprocessing/
#── __init__
#├── data_loader
#├── feature_engineer
#├── data_validator
#├── sequence_creator
#├── utils
#└── main

# data_loader.
import pandas as pd
import numpy as np
import logging
from typing import Dict, Union

# feature_engineer
from typing import List
import logging
import ta

# main.py
import logging
from pathlib import Path


# %%

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


# %%

class FeatureEngineer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def calculate_features(self, df: pd.DataFrame,missing_columns) -> pd.DataFrame:
        """Calculate all technical indicators and features"""
        try:
            df = df.copy()
            
            # only calculate features for the missing columns 
            for feature in missing_columns:
                if feature == 'SMA':
                    df['SMA'] = ta.trend.sma_indicator(df['Close'], window=20)
                elif feature == 'EMA':
                    df['EMA'] = ta.trend.ema_indicator(df['Close'], window=20)
                elif feature == 'RSI':
                    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
                elif feature == 'MACD':
                    df['MACD'] = ta.trend.macd_diff(df['Close'], window_slow=26, window_fast=12, window_sign=9)
                elif feature == 'STOCH':
                    df['STOCH'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
                elif feature == 'WILLIAMS':
                    df['WILLIAMS'] = ta.momentum.wr(df['High'], df['Low'], df['Close'], lbp=14)
                elif feature == 'CCI':
                    df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
                elif feature == 'ATR':
                    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
                elif feature == 'ADX':
                    df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
                elif feature == 'OBV':
                    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                elif feature == 'AD':
                    df['AD'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])
                elif feature == 'MA20':
                    df['MA_20'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=20).mean())
                elif feature == 'MA50':
                    df['MA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean())
                elif feature == 'MA200':
                    df['MA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean())
                elif feature == 'BB_Width_20':
                    df['BB_Width_20'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
                elif feature == 'ROC':
                    df['ROC_14'] = df.groupby('Symbol')['Close'].transform(lambda x: x.pct_change(periods=14) * 100)
                elif feature == 'Momentum':
                    df['Momentum'] = df.groupby('Symbol')['Close'].transform(lambda x: x - x.shift(10))  
                else:
                    self.logger.warning(f"Feature {feature} not recognized")              
            


            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating features: {str(e)}")
            raise
            
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize all features appropriately"""
        try:
            df = df.copy()
            
            # Price normalization
            price_cols = ['Open', 'High', 'Low', 'Close', 'MA_20', 'MA_50', 'MA_200']
            for col in price_cols:
                df[col] = df.groupby('Symbol')[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min()))
            
            # Volume normalization (log transform)
            df['Volume'] = np.log1p(df['Volume'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            raise


# %%
# sequence_creator.py
class SequenceCreator:
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.logger = logging.getLogger(__name__)
        
    def create_sequences(self, df: pd.DataFrame, 
                        feature_columns: List[str]) -> tuple[np.ndarray, np.ndarray]:
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


# %%
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
        df = data_loader.load_data('../stock_data/sp500_master_data.csv')
        
        # Calculate features
        logger.info("Calculating features...")
        required_columns=['Close', 'Returns', 'RSI_14', 'MACD', 'MA_20',  'BB_Width_20', 'ATR', 'ROC_14', 'Volume'] 
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        df = feature_engineer.calculate_features(df,missing_columns)
        
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
                feature_columns=required_columns
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



