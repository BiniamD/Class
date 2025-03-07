import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, List, Tuple
import ta

class EnhancedFinancialPreprocessor:
    def __init__(self, sequence_length: int = 20):
        """
        Initialize preprocessor with your dataset's fields plus additional features
        """
        self.sequence_length = sequence_length
        self.scalers = {}
        
        # Feature groups based on your actual data fields
        self.feature_groups = {
            'price': [
                'Open', 'High', 'Low', 'Close', 'Adj Close',
                'Returns', 'True_Range', 'ATR'
            ],
            
            'moving_averages': [
                'MA20',
                'MA50',  # Will be calculated
                'MA200', # Will be calculated
                'EMA20', # Will be calculated
            ],
            
            'volatility': [
                '20dSTD',
                'Upper_Band',
                'Lower_Band',
                'BB_Width',    # Will be calculated
                'Volatility',
                'ATR',         # Will be calculated
            ],
            
            'momentum': [
                'RSI',
                'MACD',
                'Signal_Line',
                'MACD_Histogram',  # Will be calculated
                'ROC',            # Will be calculated
                'Momentum'        # Will be calculated
            ],
            
            'volume': [
                'Volume',
                'Volume_MA',
                'Volume_Ratio',
                'OBV'            # Will be calculated
            ],
            
            'events': [
                'Dividends',
                'Stock Splits'
            ]
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional technical indicators not in original data
        """
        try:
            df = df.copy()
            
            # 1. Additional Moving Averages
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()
            df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
            
            # 2. Bollinger Band Width
            df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
            
            # 3. Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            df['True_Range'] = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = df['True_Range'].rolling(window=14).mean()
            
            # 4. MACD Histogram
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            
            # 5. Rate of Change (ROC)
            df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / 
                        df['Close'].shift(10)) * 100
            
            # 6. Momentum
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # 7. On Balance Volume (OBV)
            df['OBV'] = np.where(df['Close'] > df['Close'].shift(),
                                df['Volume'],
                                np.where(df['Close'] < df['Close'].shift(),
                                        -df['Volume'], 0)).cumsum()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating additional features: {str(e)}")
            raise
    
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features based on their characteristics
        """
        df = df.copy()
        
        # Price normalization
        price_features = self.feature_groups['price']
        self.scalers['price'] = MinMaxScaler()
        df[price_features] = self.scalers['price'].fit_transform(df[price_features])
        
        # Volume normalization (log transform then scale)
        volume_features = ['Volume', 'Volume_MA', 'OBV']
        df[volume_features] = np.log1p(df[volume_features])
        self.scalers['volume'] = MinMaxScaler()
        df[volume_features] = self.scalers['volume'].fit_transform(df[volume_features])
        
        # Note: RSI, MACD, etc. are already normalized
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, train_split: float = 0.8) -> Dict:
        """
        Complete data preparation pipeline
        """
        try:
            # 1. Ensure date is index
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.set_index('Date')
            
            # 2. Calculate additional features
            df = self.calculate_additional_features(df)
            
            # 3. Handle missing values
            df = self.handle_missing_values(df)
            
            # 4. Normalize features
            df = self.normalize_features(df)
            
            # 5. Create sequences
            X, y = self.create_sequences(df)
            
            # 6. Train/validation split
            split_idx = int(len(X) * train_split)
            
            return {
                'train': {
                    'X': X[:split_idx],
                    'y': y[:split_idx]
                },
                'val': {
                    'X': X[split_idx:],
                    'y': y[split_idx:]
                },
                'scalers': self.scalers,
                'feature_groups': self.feature_groups
            }
            
        except Exception as e:
            self.logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on feature type
        """
        df = df.copy()
        
        # Forward fill price and MA data
        price_ma_features = (self.feature_groups['price'] + 
                           self.feature_groups['moving_averages'])
        df[price_ma_features] = df[price_ma_features].fillna(method='ffill')
        
        # Zero fill volume data
        df[self.feature_groups['volume']] = df[self.feature_groups['volume']].fillna(0)
        
        # Forward fill other indicators
        df = df.fillna(method='ffill')
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for CNN-LSTM model
        """
        # Get all features except events
        features = [feat for group, feats in self.feature_groups.items() 
                   if group != 'events' for feat in feats]
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            sequence = df[features].iloc[i:(i + self.sequence_length)]
            target = df['Returns'].iloc[i + self.sequence_length]
            
            X.append(sequence.values)
            y.append(target)
            
        return np.array(X), np.array(y)

