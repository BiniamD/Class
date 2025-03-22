import yfinance as yf
from yahoo_fin import stock_info as si
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from sklearn.preprocessing import StandardScaler
import requests

class SP500DataCollector:
    def __init__(self):
        self.sp500_tickers = self._get_sp500_tickers()
        self.ALPHA_VANTAGE_API_KEY = "Y2JWGYRKAVKGXSHI"
        
    def _get_sp500_tickers(self) -> list:
        tickers = si.tickers_sp500()
        tickers = [item.replace(".", "-") for item in tickers] # Yahoo Finance uses dashes instead of dots
        return tickers
    
    def fetch_and_store_data(self, start_date: str, end_date: str, output_dir: str = 'stock_data'):
        """Fetch data for all S&P 500 companies and store in CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        master_df = pd.DataFrame()
        failed_tickers = []
        
        for ticker in self.sp500_tickers:  # For testing only
            try:
                # Fetch data
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                
                if len(df) < 100:  # Skip if not enough data
                    continue
                
                # Add ticker and company info
                df['Symbol'] = ticker
                info = stock.info
                df['Sector'] = info.get('sector', 'Unknown')
                df['Industry'] = info.get('industry', 'Unknown')
                df['Market_Cap'] = info.get('marketCap', 0)
                
                # Calculate technical indicators and features
                df = self._calculate_technical_indicators(df)
                df = self._add_fundamental_features(df, stock)
                df = self._add_market_features(df)
                
                # Add to master DataFrame
                master_df = pd.concat([master_df, df])
                print(f"Successfully processed {ticker}")
                
            except Exception as e:
                failed_tickers.append(ticker)
                print(f"Error processing {ticker}: {str(e)}")
                continue
        
        # Save master DataFrame
        master_file = os.path.join(output_dir, 'sp500_master_data.csv')
        master_df.to_csv(master_file)
        
        # Generate and save detailed summary
        summary = self._generate_detailed_summary(master_df, start_date, end_date, failed_tickers)
        
        with open(os.path.join(output_dir, 'data_summary.txt'), 'w') as f:
            for section, details in summary.items():
                f.write(f"\n{section}:\n")
                if isinstance(details, dict):
                    for key, value in details.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {details}\n")
                
        return master_file, summary
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate extensive technical indicators"""
        # Price-based indicators
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Price_Range'] = df['High'] - df['Low']
        df['Price_Range_Pct'] = df['Price_Range'] / df['Close']
        
        # Moving averages and trends
        for window in [5, 10, 20, 50, 200]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
            df[f'Returns_{window}d'] = df['Close'].pct_change(periods=window)
            
        # Volatility measures
        for window in [5, 20, 60]:
            df[f'Volatility_{window}d'] = df['Returns'].rolling(window=window).std()
            df[f'Volume_MA_{window}d'] = df['Volume'].rolling(window=window).mean()
        
        # RSI with multiple timeframes
        for window in [9, 14, 25]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = df['Close'].ewm(span=12, adjust=False).mean()
        exp26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        for window in [20, 50]:
            df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'STD_{window}d'] = df['Close'].rolling(window=window).std()
            df[f'BB_Upper_{window}'] = df[f'MA_{window}'] + (df[f'STD_{window}d'] * 2)
            df[f'BB_Lower_{window}'] = df[f'MA_{window}'] - (df[f'STD_{window}d'] * 2)
            df[f'BB_Width_{window}'] = (df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}']) / df[f'MA_{window}']
        
        # Additional technical indicators
        # Momentum
        df['Momentum_14'] = df['Close'].diff(14)
        df['ROC_14'] = df['Close'].pct_change(14) * 100
        
        # Money Flow Index (MFI)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        for window in [14, 28]:
            pos_mf = positive_flow.rolling(window=window).sum()
            neg_mf = negative_flow.rolling(window=window).sum()
            mf_ratio = pos_mf / neg_mf
            df[f'MFI_{window}'] = 100 - (100 / (1 + mf_ratio))
        
        # Volume indicators
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_StdDev'] = df['Volume'].rolling(window=20).std()
        
        # Price channels
        for window in [20, 50]:
            df[f'Upper_Channel_{window}'] = df['High'].rolling(window=window).max()
            df[f'Lower_Channel_{window}'] = df['Low'].rolling(window=window).min()
            df[f'Channel_Width_{window}'] = df[f'Upper_Channel_{window}'] - df[f'Lower_Channel_{window}']
        
        return df
    
    def _add_fundamental_features(self, df: pd.DataFrame, stock: yf.Ticker) -> pd.DataFrame:
        """Add fundamental analysis features"""
        try:
            # Get fundamental data
            info = stock.info
            
            # Add constant fundamental features
            df['PE_Ratio'] = info.get('forwardPE', np.nan)
            df['PB_Ratio'] = info.get('priceToBook', np.nan)
            df['Dividend_Yield'] = info.get('dividendYield', np.nan)
            df['Profit_Margin'] = info.get('profitMargins', np.nan)
            df['Beta'] = info.get('beta', np.nan)
            df['Enterprise_Value'] = info.get('enterpriseValue', np.nan)
            df['Forward_EPS'] = info.get('forwardEps', np.nan)
            df['Trailing_EPS'] = info.get('trailingEps', np.nan)
            
        except Exception as e:
            print(f"Error adding fundamental features: {str(e)}")
            
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-related features"""
        try:
            # Convert index to timezone-naive
            df.index = df.index.tz_localize(None)
            
            # Get S&P 500 data for the same period
            spy = yf.download('^GSPC', start=df.index[0], end=df.index[-1])
            # Convert SPY index to timezone-naive
            spy.index = spy.index.tz_localize(None)
            
            # Calculate market returns
            spy_returns = spy['Close'].pct_change()
            
            # Align indices
            df_aligned, spy_aligned = df.align(spy_returns, join='left', axis=0)
            
            # Add market features
            df['Market_Return'] = spy_aligned
            df['Market_Volatility'] = spy_aligned.rolling(window=20).std()
            
            # Calculate beta using 60-day rolling window
            df['Rolling_Beta'] = (
                df['Returns'].rolling(window=60)
                .cov(spy_aligned) / spy_aligned.rolling(window=60).var()
            )
            
            # Add VIX data if available
            try:
                vix = yf.download('^VIX', start=df.index[0], end=df.index[-1])
                vix.index = vix.index.tz_localize(None)
                _, vix_aligned = df.align(vix['Close'], join='left', axis=0)
                df['VIX'] = vix_aligned
                df['VIX_MA_10'] = vix_aligned.rolling(window=10).mean()
            except Exception as e:
                print(f"Unable to fetch VIX data: {e}")
            
        except Exception as e:
            print(f"Error adding market features: {str(e)}")
            
        return df
    
    def _generate_detailed_summary(self, df: pd.DataFrame, start_date: str, end_date: str, failed_tickers: list) -> dict:
        """Generate detailed summary statistics"""
        summary = {
            'Dataset Overview': {
                'Total Companies': len(df['Symbol'].unique()),
                'Total Observations': len(df),
                'Date Range': f"{start_date} to {end_date}",
                'Number of Features': len(df.columns),
                'Failed Tickers': len(failed_tickers),
                'Data Points per Company (Avg)': len(df) / len(df['Symbol'].unique()) if len(df['Symbol'].unique()) > 0 else 0
            },
            'Features Description': {
                'Price Indicators': [
                    'Close: Daily closing price',
                    'Returns: Daily price returns',
                    'Log_Returns: Natural logarithm of returns',
                    'Price_Range: Daily high-low range',
                    'Price_Range_Pct: Price range as percentage of closing price'
                ],
                'Moving Averages': [
                    'MA_X: Simple Moving Average (X=5,10,20,50,200 days)',
                    'EMA_X: Exponential Moving Average (X=5,10,20,50,200 days)',
                    'Returns_Xd: X-day price returns'
                ],
                'Volatility Metrics': [
                    'Volatility_Xd: X-day rolling standard deviation of returns',
                    'Volume_MA_Xd: X-day volume moving average',
                    'BB_Width_X: Bollinger Band width for X-day period'
                ],
                'Technical Indicators': [
                    'RSI_X: Relative Strength Index (X=9,14,25 days)',
                    'MACD: Moving Average Convergence Divergence',
                    'Signal_Line: MACD signal line',
                    'MACD_Histogram: MACD - Signal Line',
                    'Momentum_14: 14-day momentum',
                    'ROC_14: 14-day rate of change',
                    'MFI_X: Money Flow Index (X=14,28 days)',
                    'Channel_Width_X: Price channel width (X=20,50 days)'
                ],
                'Volume Indicators': [
                    'OBV: On-Balance Volume',
                    'Volume_Ratio: Volume relative to 20-day average',
                    'Volume_StdDev: 20-day volume standard deviation'
                ],
                'Fundamental Features': [
                    'PE_Ratio: Price to Earnings ratio',
                    'PB_Ratio: Price to Book ratio',
                    'Dividend_Yield: Annual dividend yield',
                    'Profit_Margin: Company profit margin',
                    'Beta: Stock beta coefficient',
                    'Enterprise_Value: Company enterprise value',
                    'Forward_EPS: Forward earnings per share',
                    'Trailing_EPS: Trailing earnings per share'
                ],
                'Market Features': [
                    'Market_Return: S&P 500 daily returns',
                    'Market_Volatility: S&P 500 20-day volatility',
                    'Rolling_Beta: 60-day rolling beta coefficient',
                    'VIX: CBOE Volatility Index',
                    'VIX_MA_10: 10-day moving average of VIX'
                ]
            },
            'Data Quality': {
                'Missing Values (%)': df.isnull().mean().mean() * 100,
                'Memory Usage (MB)': df.memory_usage().sum() / 1024**2
            }
        }
        
        return summary

def main():
    # Initialize collector
    collector = SP500DataCollector()
    
    # Set date range for 5 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Fetch and store data
    master_file, summary = collector.fetch_and_store_data(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    print("\nData Collection Summary:")
    for section, details in summary.items():
        print(f"\n{section}:")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {details}")

if __name__ == "__main__":
    main()