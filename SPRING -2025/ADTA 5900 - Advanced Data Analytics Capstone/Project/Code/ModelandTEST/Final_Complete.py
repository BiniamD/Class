# %%
import pandas as pd
import os
import numpy as np
import json
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, Activation, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

#import warnings
import warnings
warnings.filterwarnings('ignore')


# %%
# Technical Indicator Functions
def calculate_moving_averages(df, windows=[50, 200]):
    """Calculate moving averages for specified windows."""
    for window in windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    short_ema = df['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACD'] = short_ema - long_ema
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df

def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands."""
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=window).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=window).std()
    return df


# %%
# Feature Selection with Mutual Information
def categories_features(df):
    # Define feature categories
    price_indicators = ['Close', 'Returns', 'Log_Returns', 'Price_Range', 'Price_Range_Pct']
    moving_averages = [col for col in df.columns if col.startswith(('MA_', 'EMA_', 'Returns_'))]
    volatility_metrics = [col for col in df.columns if col.startswith(('Volatility_', 'Volume_MA_', 'BB_Width_'))]
    technical_indicators = ['RSI_9', 'RSI_14', 'RSI_25', 'MACD', 'Signal_Line', 'MACD_Histogram',
                        'Momentum_14', 'ROC_14', 'MFI_14', 'MFI_28'] + \
                        [col for col in df.columns if col.startswith('Channel_Width_')]
    volume_indicators = ['OBV', 'Volume_Ratio', 'Volume_StdDev']
    fundamental_features = ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Profit_Margin', 'Beta', 
                        'Enterprise_Value', 'Forward_EPS', 'Trailing_EPS']
    market_features = ['Market_Return', 'Market_Volatility', 'Rolling_Beta', 'VIX', 'VIX_MA_10']
    all_features = (price_indicators + moving_averages + volatility_metrics +
                technical_indicators + volume_indicators + fundamental_features + market_features)
    features = [f for f in all_features if f in df.columns]
    
    return features
# Function to select features using mutual information
def select_features_with_mi(df, features, target_col='Target', n_select=30):

    """Select top features using mutual information."""
    data = df.dropna(subset=[target_col])
    mi_scores = mutual_info_classif(data[features], data[target_col])
    mi_df = pd.DataFrame({'Feature': features, 'MI Score': mi_scores})
    mi_df = mi_df.sort_values('MI Score', ascending=False)
    print("Top 15 features by mutual information:")
    print(mi_df.head(15))
    return mi_df.head(n_select)['Feature'].tolist()
# Function to load and preprocess data
def load_and_preprocess_data(file_path, symbol, seq_length):
    """
    Load stock data from a CSV file, preprocess it, and create sequences for the specified symbol.
    
    Args:
        file_path (str): Path to the CSV file containing stock data.
        symbol (str): Stock symbol to filter (e.g., 'AAPL').
        seq_length (int): Number of time steps in each input sequence.
    
    Returns:
        np.array: Input sequences (X).
        np.array: Target values (y).
        list: List of feature column names.
    """
    # Load data
    df = pd.read_csv(file_path)
    if 'Symbol' in df.columns:
        df = df[df['Symbol'] == symbol].copy()
    else:
        raise ValueError(f"No data found for symbol '{symbol}' in the dataset.")
    
    # Convert 'Date' to datetime and set as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Calculate technical indicators
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)

    # Handle missing values with forward fill
    df.ffill(inplace=True)
    
    # Define target: 1 if next day's return > 0, else 0 include sell if
    df['Return'] = df['Close'].pct_change().shift(-1)
    df['Target'] = np.where(df['Return'] > 0, 1, 0)
    df = df.dropna()
    
    # Define features
    features = categories_features(df)
    # Feature selection
    #selected_features = select_features_with_mi(df, features, 'Target', n_select=15)
    selected_features = ['Close', 'Volume', 'Return', 'MA_50', 'MA_200', 'RSI', 'MACD', 'Signal_Line', 'BB_Middle', 'BB_Upper', 'BB_Lower']
    print(f"Selected {len(selected_features)} features")
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[selected_features])
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i])
        y.append(df['Target'].iloc[i])
    X, y = np.array(X), np.array(y)
    
    return X, y, features, df

def build_cnn_bilstm_model(seq_length, num_features, num_classes=1):
    """
    Build a CNN-BiLSTM model with attention for time series forecasting in a trading system.
    
    Args:
        seq_length (int): Number of time steps in each input sequence.
        num_features (int): Number of features in the input data.
        num_classes (int): Number of output classes (default: 1 for binary classification).
    
    Returns:
        Model: Compiled Keras model.
    """
    # Define input layer
    inputs = Input(shape=(seq_length, num_features))
    
    # CNN layers for feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # First BiLSTM layer for temporal dependencies
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    
    # Second BiLSTM layer, outputting a sequence for attention
    sequence = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Dropout(0.2)(sequence)

    # Final BiLSTM layer
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.2)(x)

    # Attention mechanism using Keras operations
    attention_scores = Dense(1)(sequence)  # Shape: (batch_size, time_steps, 1)
    attention_weights = Activation('softmax')(attention_scores)  # Normalize weights across time steps
    #attention_output = Lambda(lambda x: tf.reduce_sum(x[0] * x[1], axis=1))([sequence, attention_weights])  # Weighted sum: (batch_size, 64)
    # Update inside build_cnn_bilstm_model where Lambda is used:
    attention_output = Lambda(lambda x: tf.reduce_sum(x[0] * tf.expand_dims(tf.squeeze(x[1], -1), -1), axis=1))([sequence, attention_weights])
    # Dense layers for prediction
    x = Dense(32, activation='relu')(attention_output)
    outputs = Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
    
    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split the data into training, validation, and test sets.
    
    Args:
        X (np.array): Input sequences.
        y (np.array): Target values.
        train_ratio (float): Proportion of data for training.
        val_ratio (float): Proportion of data for validation.
    
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test, train_size, val_size
    """
    num_samples = len(X)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
    return X_train, X_val, X_test, y_train, y_val, y_test, train_size, val_size

# Function to evaluate and visualize model performance
def evaluate_and_visualize_model(y_test_labels, y_pred, history, symbol, symbol_dir):
    """
    Evaluate the model's performance and visualize the results for a given stock symbol.

    Args:
        y_test_labels (array-like): True labels for the test set.
        y_pred (array-like): Predicted labels for the test set.
        history (History): Training history object from model.fit, containing accuracy and loss metrics.
        symbol (str): Stock symbol (e.g., 'AAPL') used for naming plots.
        symbol_dir (str): Directory path where the plots will be saved.

    Returns:
        None: Prints metrics and saves plots to the specified directory.
    """
    # Calculate performance metrics
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='weighted')
    recall = recall_score(y_test_labels, y_pred, average='weighted')
    f1 = f1_score(y_test_labels, y_pred, average='weighted')

    # Print metrics and classification report
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hold', 'Buy'], yticklabels=['Hold', 'Buy'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {symbol}')
    plt.savefig(os.path.join(symbol_dir, f'confusion_matrix_{symbol}.png'))
    plt.close()

    # Plot and save training history (accuracy and loss)
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {symbol}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {symbol}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(symbol_dir, f'training_history_{symbol}.png'))
    plt.close()
    return accuracy, precision, recall, f1
# Placeholder for advanced trading signal generation
def generate_advanced_trading_signals(model, X_test, df_test, confidence_threshold):
    """Generate trading signals based on model predictions."""
    y_pred_proba = model.predict(X_test)
    
    # Ensure proper shape for binary classification
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 1:
        y_pred_proba = y_pred_proba.flatten()
    
    signals = pd.DataFrame(index=df_test.index[:len(y_pred_proba)])
    
    # For binary classification (0 = down, 1 = up)
    signals['pred_prob'] = y_pred_proba  
    signals['predicted_class'] = (signals['pred_prob'] > confidence_threshold).astype(int)
    signals['confidence'] = np.where(signals['pred_prob'] > 0.5, 
                                    signals['pred_prob'], 
                                    1 - signals['pred_prob'])
    
    # Generate positions (-1 for short, 0 for neutral, 1 for long)
    signals['position'] = 0
    signals.loc[(signals['predicted_class'] == 1) & (signals['confidence'] > confidence_threshold), 'position'] = 1  # Long
    signals.loc[(signals['predicted_class'] == 0) & (signals['confidence'] > confidence_threshold), 'position'] = -1  # Short
    
    # Add trading signals label
    signals['Signal'] = 'Hold'  # Default
    signals.loc[signals['position'] == 1, 'Signal'] = 'Buy'
    signals.loc[signals['position'] == -1, 'Signal'] = 'Sell'
    
    # Calculate position size based on confidence
    signals['position_size'] = signals['position'] * (signals['confidence'] - 0.5) * 2
    signals.loc[signals['position_size'] < 0, 'position_size'] = signals['position_size'].abs()
    
    # Add market data
    signals['price'] = df_test['Close'].values[:len(signals)]
    
    # Calculate returns
    signals['market_return'] = np.log(signals['price'] / signals['price'].shift(1))
    signals['strategy_return'] = signals['position'].shift(1) * signals['market_return']
    signals['sized_strategy_return'] = signals['position_size'].shift(1) * signals['market_return']
    
    # Handle NaN values in return calculations
    signals.dropna(subset=['market_return', 'strategy_return', 'sized_strategy_return'], inplace=True)
    
    # Calculate cumulative returns and drawdowns
    signals['cumulative_market_return'] = np.exp(signals['market_return'].cumsum()) - 1
    signals['cumulative_strategy_return'] = np.exp(signals['sized_strategy_return'].cumsum()) - 1
    signals['drawdown'] = signals['cumulative_strategy_return'] - signals['cumulative_strategy_return'].cummax()
    
    # Calculate performance metrics
    if len(signals) > 0:
        total_return = np.exp(signals['sized_strategy_return'].sum()) - 1
        annual_return = np.exp(signals['sized_strategy_return'].mean() * 252) - 1
        sharpe_ratio = np.sqrt(252) * signals['sized_strategy_return'].mean() / (signals['sized_strategy_return'].std() or 1e-8)
        max_drawdown = signals['drawdown'].min()
        win_rate = len(signals[signals['sized_strategy_return'] > 0]) / (len(signals[signals['sized_strategy_return'] != 0]) or 1)
        gross_profits = signals.loc[signals['sized_strategy_return'] > 0, 'sized_strategy_return'].sum()
        gross_losses = abs(signals.loc[signals['sized_strategy_return'] < 0, 'sized_strategy_return'].sum())
        profit_factor = gross_profits / (gross_losses or 1e-8)
        
        performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    else:
        performance_metrics = {
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0
        }
    
    return signals, performance_metrics
# Function to visualize trading performance
def visualize_trading_performance(signals, performance_metrics, symbol_name,symbol_dir):
    """Visualize trading performance metrics and signals."""
    fig = plt.figure(figsize=(15, 12))
    ax1 = fig.add_subplot(3, 1, 1)
    signals['cumulative_market_return'].plot(ax=ax1, label=f'{symbol_name} Return', color='blue', alpha=0.7)
    signals['cumulative_strategy_return'].plot(ax=ax1, label='Strategy Return', color='green')
    ax1.set_title(f'Cumulative Returns Comparison - {symbol_name}')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True)
    ax2 = fig.add_subplot(3, 1, 2)
    signals['drawdown'].plot(ax=ax2, color='red')
    ax2.set_title('Strategy Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(signals.index, signals['price'], color='black', alpha=0.7)
    buy_signals = signals[signals['position'].diff() > 0]
    sell_signals = signals[signals['position'].diff() < 0]
    ax3.scatter(buy_signals.index, buy_signals['price'], marker='^', color='green', s=100, label='Buy')
    ax3.scatter(sell_signals.index, sell_signals['price'], marker='v', color='red', s=100, label='Sell')
    ax3.set_title(f'Trading Signals - {symbol_name}')
    ax3.set_ylabel('Price')
    ax3.legend()
    plt.figtext(0.01, 0.01, f"""
    {symbol_name} Performance Metrics:
    - Total Return: {performance_metrics['total_return']*100:.2f}%
    - Annual Return: {performance_metrics['annual_return']*100:.2f}%
    - Sharpe Ratio: {performance_metrics['sharpe_ratio']:.2f}
    - Max Drawdown: {performance_metrics['max_drawdown']*100:.2f}%
    - Win Rate: {performance_metrics['win_rate']*100:.2f}%
    - Profit Factor: {performance_metrics['profit_factor']:.2f}
    """, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    #plt.savefig(f'{symbol_name}_trading_performance.png')
    plt.savefig(os.path.join(symbol_dir, f'{symbol_name}_trading_performance.png'))
    plt.close()
# Function to save trading signals and performance metrics    
def simulate_trades(signals, df, stop_loss_pct=0.02, take_profit_pct=0.05, max_holding_days=30):
    """Simulate trades with stop-loss and take-profit."""
    portfolio = 10000  # Initial capital
    position = 0  # 0: no position, 1: long position
    entry_price = 0
    days_held = 0
    trade_returns = []
    portfolio_values = [portfolio]

    # Ensure indices are aligned
    if isinstance(signals.index, pd.DatetimeIndex) and isinstance(df.index, pd.DatetimeIndex):
        common_index = signals.index.intersection(df.index)
        signals = signals.loc[common_index]
        df = df.loc[common_index]

    for date, row in signals.iterrows():
        try:
            signal = row['position']
            close_price = df.loc[date, 'Close'] if date in df.index else None
            
            if close_price is None:
                continue
                
            if position == 0:  # No position
                if signal == 1:  # Buy signal
                    position = 1
                    entry_price = close_price
                    days_held = 0
                    print(f"Entering position at {close_price} on {date}")
            elif position == 1:  # Holding position
                days_held += 1
                if close_price <= entry_price * (1 - stop_loss_pct):
                    position = 0
                    trade_return = (close_price - entry_price) / entry_price
                    portfolio *= (1 + trade_return)
                    trade_returns.append(trade_return)
                    print(f"Stop-loss triggered at {close_price} on {date}, return: {trade_return:.2%}")
                elif close_price >= entry_price * (1 + take_profit_pct):
                    position = 0
                    trade_return = (close_price - entry_price) / entry_price
                    portfolio *= (1 + trade_return)
                    trade_returns.append(trade_return)
                    print(f"Take-profit triggered at {close_price} on {date}, return: {trade_return:.2%}")
                elif days_held >= max_holding_days:
                    position = 0
                    trade_return = (close_price - entry_price) / entry_price
                    portfolio *= (1 + trade_return)
                    trade_returns.append(trade_return)
                    print(f"Max holding period reached, exiting at {close_price} on {date}, return: {trade_return:.2%}")

            portfolio_values.append(portfolio)
            
        except Exception as e:
            print(f"Error processing trade on {date}: {e}")
            continue

    # Calculate performance metrics
    daily_returns = pd.Series(portfolio_values).pct_change().dropna()
    total_return = (portfolio - 10000) / 10000
    sharpe_ratio = daily_returns.mean() / (daily_returns.std() or 1e-8) * np.sqrt(252)
    max_drawdown = (pd.Series(portfolio_values).cummax() - pd.Series(portfolio_values)).max() / pd.Series(portfolio_values).cummax().max()

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'num_trades': len(trade_returns),
        'win_rate': sum(r > 0 for r in trade_returns) / (len(trade_returns) or 1)
    }
# Main function to run the trading system
def run_trading_system(file_path, symbol='AAPL', seq_length=30, confidence_threshold=0.6,results_dir='results'):
    """
    Run the trading system for a given stock symbol using a CNN-BiLSTM model.
    
    Args:
        file_path (str): Path to the CSV file with stock data.
        symbol (str): Stock symbol (default: 'AAPL').
        seq_length (int): Number of days in each sequence (default: 30).
        confidence_threshold (float): Threshold for generating buy signals (default: 0.6).
    
    Returns:
        model: Trained CNN-BiLSTM model.
        signals (pd.DataFrame): Generated trading signals.
        performance_metrics (dict): Performance metrics including accuracy, precision, recall, and F1 score.
    """
    # Create symbol-specific directory
    symbol_dir = os.path.join(results_dir, symbol)
    os.makedirs(symbol_dir, exist_ok=True)
    # Load and preprocess data
    X, y, feature_cols, df = load_and_preprocess_data(file_path, symbol, seq_length)
    
    # Split data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test, train_size, val_size = split_data(X, y)
    
    # Reshape X_train for SMOTE (from 3D to 2D)
    num_samples, seq_len, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(num_samples, seq_len * num_features)

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_reshaped, y_train)
    
    # Reshape X_train_resampled back to 3D
    new_num_samples = X_train_resampled.shape[0]
    X_train_resampled = X_train_resampled.reshape(new_num_samples, seq_length, num_features)

    # Build the CNN-BiLSTM model
    #num_features = X.shape[2]
    model = build_cnn_bilstm_model(seq_length, num_features)
    #model = build_improved_cnn_bilstm_model(seq_length, num_features)
    
    # Build and train the model with resampled data
    model = build_cnn_bilstm_model(seq_length, num_features)
    history = model.fit(
        X_train_resampled, y_train_resampled,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > confidence_threshold).astype(int)
    y_test_labels = y_test  # Binary classification, no argmax needed

    # Plot training and validation loss
    accuracy, precision, recall, f1 = evaluate_and_visualize_model(y_test_labels, y_pred, history, symbol, symbol_dir)
    
    # Generate trading signals
    print(f"Generating trading signals for {symbol}...")
    start_index = seq_length + train_size + val_size
    df_test = df.iloc[start_index:start_index + len(y_test)]
    signals, performance_metrics = generate_advanced_trading_signals(
        model, 
        X_test, 
        df_test, 
        confidence_threshold
    )
    
    # Visualize trading performance (placeholder)
    visualize_trading_performance(signals, performance_metrics, symbol, symbol_dir)
    
    # Simulate trades
    trades = simulate_trades(signals, df)

    # Update performance metrics with evaluation results
    performance_metrics.update({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    with open(os.path.join(symbol_dir, f'metrics_{symbol}.json'), 'w') as f:
        json.dump(performance_metrics, f, indent=4)

    # Simulate trades
    #Save signals and trades to CSV
    signals.to_csv(os.path.join(symbol_dir, f'signals_{symbol}.csv'))
    pd.DataFrame([trades]).to_csv(os.path.join(symbol_dir, f'trades_{symbol}.csv'), index=False)
    
    return model, signals, performance_metrics, trades


# %%
# List of top 20 stock symbols
top_20_symbols = ['AAPL', 'NVDA', 'MSFT', 'AMZN', 'META', 'GOOGL', 'AVGO', 'TSLA',
                  'BRK.B', 'GOOG', 'JPM', 'LLY', 'V', 'COST', 'MA', 'UNH',
                  'NFLX', 'WMT', 'PG', 'JNJ', 'HD', 'ABBV', 'BAC', 'CRM']
#top_20_symbols = ['NVDA']
# Example usage
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
if __name__ == "__main__":
    for symbol in top_20_symbols:
        print(f"Processing {symbol}...")
        try:
            model, signals, performance,trades = run_trading_system(
                file_path='sp500_master_data.csv',
                symbol=symbol,
                seq_length=30,
                confidence_threshold=0.6,
                results_dir=results_dir
            )
            print(f"\nPerformance Metrics {symbol}:", performance)
            print(f"\nTrades {symbol}:", trades)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

# %%
import pandas as pd
import json
# List of top 20 stock symbols
top_20_symbols = ['AAPL', 'NVDA', 'MSFT', 'META', 'GOOGL', 'AVGO', 
                   'GOOG', 'JPM', 'LLY', 'V', 'COST', 'MA', 'UNH',
                   'WMT', 'PG', 'JNJ', 'HD', 'ABBV', 'BAC', 'CRM','MMM','T']
#top_20_symbols = ['T']
# load the data with symbol as the frist column in dataframe from the results folder
def load_data(symbol):
    symbol_dir = f'results/{symbol}'
    signals = pd.read_csv(f'{symbol_dir}/signals_{symbol}.csv', index_col=0)
    trades = pd.read_csv(f'{symbol_dir}/trades_{symbol}.csv')
    with open(f'{symbol_dir}/metrics_{symbol}.json', 'r') as f:
        metrics = json.load(f)
    return signals, trades, metrics

# Load data for each symbol and store to dataframe
data = []
all_trades = []
all_signals = []
for symbol in top_20_symbols:
    signals, trades, metrics = load_data(symbol)
    data.append({
        'Symbol': symbol,
        'Total Return': metrics['total_return'],
        'Annual Return': metrics['annual_return'],
        'Sharpe Ratio': metrics['sharpe_ratio'],
        'Max Drawdown': metrics['max_drawdown'],
        'Win Rate': metrics['win_rate'],
        'Profit Factor': metrics['profit_factor']
    })
      # Add symbol column to trades dataframe
    trades['Symbol'] = symbol
    
    # Reorder columns to have Symbol first
    cols = trades.columns.tolist()
    cols.remove('Symbol')
    cols = ['Symbol'] + cols
    trades = trades[cols]
    
    #print(trades)
    all_trades.append(trades)

    # Add symbol column to trades dataframe
    signals['Symbol'] = symbol
    
    # Reorder columns to have Symbol first
    cols = signals.columns.tolist()
    cols.remove('Symbol')
    cols = ['Symbol'] + cols
    signals = signals[cols]
    
    #print(trades)
    all_signals.append(signals)

# Create a DataFrame from the list of dictionaries for metrics
df_metrics = pd.DataFrame(data)
#print(df_metrics)
#save to file
df_metrics.to_csv('metrics.csv', index=False)

# Combine all trades data
combined_trades = pd.concat(all_trades, ignore_index=True)
#create df for trades
df_trades = pd.DataFrame(combined_trades)
#save to file
df_trades.to_csv('trades.csv', index=False)

# Combine all signals data
#print("\nAll trades with Symbol column:")
#print(df_trades)
# Combine all signals data
combined_signals = pd.concat(all_signals, ignore_index=True)
#create df for signals
df_signals = pd.DataFrame(combined_signals)
#save to file
df_signals.to_csv('signals.csv', index=False)
#print("\nAll trades with Symbol column:")
#print(df_signals)

# %%
# pick only win trades
df_metrics_win = df_metrics[df_metrics['Win Rate'] > 0.5]
df_metrics_win = df_metrics_win.sort_values('Total Return', ascending=False)

# pick only win trades

# what is overall performance of the trading system
# Calculate overall performance metrics
total_return = df_metrics_win['Total Return'].mean()
annual_return = df_metrics_win['Annual Return'].mean()
sharpe_ratio = df_metrics_win['Sharpe Ratio'].mean()
max_drawdown = df_metrics_win['Max Drawdown'].max()
win_rate = df_metrics_win['Win Rate'].mean()
profit_factor = df_metrics_win['Profit Factor'].mean()

# Print overall performance metrics
print("\nOverall Performance Metrics:")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Return: {annual_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")

# %%
# what is overall performance of the trading system
# Calculate overall performance metrics
total_return = df_metrics['Total Return'].mean()
annual_return = df_metrics['Annual Return'].mean()
sharpe_ratio = df_metrics['Sharpe Ratio'].mean()
max_drawdown = df_metrics['Max Drawdown'].max()
win_rate = df_metrics['Win Rate'].mean()
profit_factor = df_metrics['Profit Factor'].mean()

# Print overall performance metrics
print("\nOverall Performance Metrics:")
print(f"Total Return: {total_return:.2%}")
print(f"Annual Return: {annual_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")
#save to file


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# Set the style for all plots
plt.style.use('fivethirtyeight')
sns.set_palette('Set2')

# Create a directory for saving visualizations
import os
os.makedirs('visualizations', exist_ok=True)


# %%
# 1. VISUALIZING df_metrics DATA
#=================================

# First, reset the index if Symbol is set as the index
if 'Symbol' not in df_metrics.columns and df_metrics.index.name == 'Symbol':
    df_metrics = df_metrics.reset_index()

# 1.1 - Performance Metrics Heatmap
plt.figure(figsize=(14, 10))
# Sort by Total Return
df_metrics_sorted = df_metrics.sort_values('Total Return', ascending=False)
# Set Symbol as index for the heatmap if it's not already
if 'Symbol' in df_metrics_sorted.columns:
    df_metrics_sorted = df_metrics_sorted.set_index('Symbol')
# Create a heatmap
metrics_heatmap = sns.heatmap(df_metrics_sorted, annot=True, fmt='.3f', cmap='RdYlGn', linewidths=0.5)
plt.title('Performance Metrics Heatmap (Sorted by Total Return)', fontsize=16)
plt.tight_layout()
plt.savefig('visualizations/1_1metrics_heatmap.png', dpi=300)
plt.close()

# For the next visualizations, make sure to use Symbol properly
# First get Symbol back as a column if it's an index
if 'Symbol' not in df_metrics.columns:
    df_metrics = df_metrics.reset_index()

# 1.2 - Top 10 Symbols by Total Return
plt.figure(figsize=(12, 6))
top_symbols = df_metrics.sort_values('Total Return', ascending=False).head(10)
ax = sns.barplot(x='Symbol', y='Total Return', data=top_symbols)
plt.title('Top 10 Symbols by Total Return', fontsize=16)
plt.xlabel('Symbol')
plt.ylabel('Total Return')
plt.xticks(rotation=45)
# Add value labels on top of bars
for i, v in enumerate(top_symbols['Total Return']):
    ax.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('visualizations/1_2top_symbols_by_return.png', dpi=300)
plt.close()

# 1.3 - Risk-Return Scatter Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    df_metrics['Max Drawdown'], 
    df_metrics['Total Return'], 
    s=df_metrics['Sharpe Ratio']*500, 
    c=df_metrics['Win Rate'],
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(scatter, label='Win Rate')
plt.title('Risk-Return Analysis with Sharpe Ratio and Win Rate', fontsize=16)
plt.xlabel('Risk (Max Drawdown)')
plt.ylabel('Total Return')

# Add labels for each symbol
for i, symbol in enumerate(df_metrics.index):
    plt.annotate(
        symbol, 
        (df_metrics['Max Drawdown'].iloc[i], df_metrics['Total Return'].iloc[i]),
        textcoords="offset points",
        xytext=(0,10), 
        ha='center'
    )

plt.tight_layout()
plt.savefig('visualizations/1_3risk_return_scatter.png', dpi=300)
plt.close()


# 2. VISUALIZING df_trades DATA
#===============================

# 2.1 - Number of Trades by Symbol
plt.figure(figsize=(14, 7))
trade_counts = df_trades.groupby('Symbol')['num_trades'].sum().sort_values(ascending=False)
ax = sns.barplot(x=trade_counts.index, y=trade_counts.values)
plt.title('Number of Trades by Symbol', fontsize=16)
plt.xlabel('Symbol')
plt.ylabel('Number of Trades')
plt.xticks(rotation=45)
# Add value labels on top of bars
for i, v in enumerate(trade_counts.values):
    ax.text(i, v + 0.1, f'{int(v)}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('visualizations/2_1trades_by_symbol.png', dpi=300)
plt.close()

# 2.2 - Win Rate vs. Total Return
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_trades, x='win_rate', y='total_return', hue='Symbol', size='num_trades', 
                sizes=(50, 500), alpha=0.7)
plt.title('Win Rate vs. Total Return', fontsize=16)
plt.xlabel('Win Rate')
plt.ylabel('Total Return')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/2_2win_rate_vs_return.png', dpi=300)
plt.close()

# 2.3 - Combined Performance Dashboard
plt.figure(figsize=(20, 16))
gs = GridSpec(3, 2)

# Return Distribution
ax1 = plt.subplot(gs[0, 0])
sns.histplot(df_trades['total_return'], kde=True, ax=ax1)
ax1.set_title('Distribution of Total Returns')
ax1.set_xlabel('Total Return')

# Sharpe Ratio vs Max Drawdown
ax2 = plt.subplot(gs[0, 1])
sns.scatterplot(data=df_trades, x='max_drawdown', y='sharpe_ratio', size='num_trades',
                sizes=(50, 500), alpha=0.7, ax=ax2)
ax2.set_title('Sharpe Ratio vs Max Drawdown')
ax2.set_xlabel('Max Drawdown')
ax2.set_ylabel('Sharpe Ratio')

# Win Rate Distribution
ax3 = plt.subplot(gs[1, 0])
sns.boxplot(data=df_trades, y='win_rate', x='Symbol', ax=ax3)
ax3.set_title('Win Rate Distribution by Symbol')
ax3.set_ylabel('Win Rate')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)

# Trade Count vs Sharpe Ratio
ax4 = plt.subplot(gs[1, 1])
sns.regplot(data=df_trades, x='num_trades', y='sharpe_ratio', ax=ax4, scatter_kws={'alpha':0.5})
ax4.set_title('Number of Trades vs Sharpe Ratio')
ax4.set_xlabel('Number of Trades')
ax4.set_ylabel('Sharpe Ratio')

# Top 5 and Bottom 5 Performers
ax5 = plt.subplot(gs[2, :])
top5 = df_trades.sort_values('total_return', ascending=False).head(5)
bottom5 = df_trades.sort_values('total_return').head(5)
compare = pd.concat([top5, bottom5])
compare_melted = pd.melt(compare.reset_index(), id_vars=['Symbol'], 
                          value_vars=['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'])
sns.barplot(data=compare_melted, x='Symbol', y='value', hue='variable', ax=ax5)
ax5.set_title('Comparison of Top 5 and Bottom 5 Performers')
ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
ax5.legend(title='Metric')

plt.suptitle('Trading Performance Dashboard', fontsize=20, y=0.98)
plt.tight_layout()
plt.savefig('visualizations/2_3performance_dashboard.png', dpi=300)
plt.close()

# 2.4 - Sharpe Ratio Dashboard
# Filter for top 10 sharpe ratios
top_sharpe = df_trades.sort_values('sharpe_ratio', ascending=False).head(10)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='Symbol', y='sharpe_ratio', data=top_sharpe)
plt.title('Top 10 Symbols by Sharpe Ratio', fontsize=16)
#plt.xlabel('Symbol')
plt.ylabel('Sharpe Ratio')
plt.xticks(rotation=45)
# Add value labels on top of bars
for i, v in enumerate(top_sharpe['sharpe_ratio']):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('visualizations/2_4top_symbols_by_sharpe.png', dpi=300)
plt.close()

print("Visualizations have been created and saved to the 'visualizations' folder.")

# %%



