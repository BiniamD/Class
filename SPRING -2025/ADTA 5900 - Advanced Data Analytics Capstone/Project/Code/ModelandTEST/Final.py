# %%
# %%
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import linregress

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, MaxPooling1D, LSTM, Bidirectional, add
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Multiply, Lambda, RepeatVector, Permute
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# %%
# 1. Enhanced Data Preprocessing
def preprocess_stock_data(df):
    """Comprehensive preprocessing for stock data"""
    # 1. Forward-fill and backward-fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)  # Any remaining NaNs
    
    # 2. Create target variable with adaptive threshold
    # Use a fixed threshold for simplicity in this implementation
    threshold = 0.003  # Can be adjusted based on market volatility
    df['target'] = np.where(df['Returns'].shift(-1) > threshold, 2,
                           np.where(df['Returns'].shift(-1) < -threshold, 0, 1))
    
    # Remove rows with NaN target
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)
    
    return df

# 2. Feature Selection with Mutual Information
def select_features_with_mi(df, features, target_col='target', n_select=30):
    """Select features using mutual information"""
    # Drop rows with missing target values
    data = df.dropna(subset=[target_col])
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(data[features], data[target_col])
    
    # Create DataFrame of features and scores
    mi_df = pd.DataFrame({'Feature': features, 'MI Score': mi_scores})
    mi_df = mi_df.sort_values('MI Score', ascending=False)
    
    # Print top features and scores
    print("Top 15 features by mutual information:")
    print(mi_df.head(15))
    
    # Return top n_select features
    return mi_df.head(n_select)['Feature'].tolist()

# 3. Create correlation tensor features
def create_correlation_tensor(df, technical_indicators, window_size=30):
    """Create correlation tensors from technical indicators"""
    n_samples = len(df) - window_size + 1
    n_features = len(technical_indicators)
    correlation_tensor = np.zeros((n_samples, n_features, n_features))
    
    for i in range(n_samples):
        window_data = df.iloc[i:i+window_size][technical_indicators]
        correlation_matrix = window_data.corr().fillna(0).values
        correlation_tensor[i] = correlation_matrix
    
    return correlation_tensor

# 4. Sequence data preparation
def prepare_sequence_data(df, sequence_length, target_column, feature_columns):
    """Prepare sequence data for time series forecasting"""
    X = []
    y = []
    
    # Extract features and target
    data = df[feature_columns].values
    targets = df[target_column].values
    
    # Create sequences
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(targets[i+sequence_length])
    
    return np.array(X), np.array(y)


# %%
# 5. Build model with parameters for hyperparameter tuning
def build_model_with_params(input_shape, filters, lstm_units, dropout_rates, learning_rate, num_classes=3):
    """Build model with configurable parameters for hyperparameter tuning"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Block 1
    conv1 = Conv1D(filters=filters[0], kernel_size=3, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling1D(pool_size=2)(act1)
    drop1 = Dropout(dropout_rates[0])(pool1)
    
    # CNN Block 2
    conv2 = Conv1D(filters=filters[1], kernel_size=3, padding='same')(drop1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    pool2 = MaxPooling1D(pool_size=2)(act2)
    drop2 = Dropout(dropout_rates[1])(pool2)
    
    # Bidirectional LSTM layers
    lstm1 = Bidirectional(LSTM(units=lstm_units[0], return_sequences=True))(drop2)
    drop_lstm1 = Dropout(dropout_rates[2])(lstm1)
    lstm2 = Bidirectional(LSTM(units=lstm_units[1], return_sequences=False))(drop_lstm1)
    drop_lstm2 = Dropout(dropout_rates[2])(lstm2)
    
    # Final dense layers
    dense1 = Dense(64, activation='relu')(drop_lstm2)
    drop_dense = Dropout(dropout_rates[2])(dense1)
    outputs = Dense(num_classes, activation='softmax')(drop_dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a lower learning rate
    optimizer = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Enhanced CNN-BiLSTM Model with Attention
def build_enhanced_cnn_bilstm_model(input_shape, num_classes=3):
    """Build enhanced CNN-BiLSTM model with attention and residual connections"""
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Block 1
    conv1 = Conv1D(filters=128, kernel_size=3, padding='same')(inputs)
    bn1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(bn1)
    pool1 = MaxPooling1D(pool_size=2)(act1)
    drop1 = Dropout(0.2)(pool1)
    
    # CNN Block 2
    conv2 = Conv1D(filters=256, kernel_size=3, padding='same')(drop1)
    bn2 = BatchNormalization()(conv2)
    act2 = Activation('relu')(bn2)
    pool2 = MaxPooling1D(pool_size=2)(act2)
    drop2 = Dropout(0.3)(pool2)
    
    # Bidirectional LSTM layers
    lstm1 = Bidirectional(LSTM(units=128, return_sequences=True))(drop2)
    drop_lstm1 = Dropout(0.4)(lstm1)
    lstm2 = Bidirectional(LSTM(units=64, return_sequences=False))(drop_lstm1)
    drop_lstm2 = Dropout(0.4)(lstm2)
    
    # Final dense layers
    dense1 = Dense(64, activation='relu')(drop_lstm2)
    drop_dense = Dropout(0.5)(dense1)
    outputs = Dense(num_classes, activation='softmax')(drop_dense)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use a lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# %%

# 6. Dynamic Class Weight Callback
class DynamicClassWeightCallback(tf.keras.callbacks.Callback):
    """Callback to dynamically adjust class weights during training"""
    def __init__(self, training_data, training_targets, initial_weights=None, adjustment_factor=0.05):
        super(DynamicClassWeightCallback, self).__init__()
        self.X_train = training_data
        self.y_train = training_targets
        self.adjustment_factor = adjustment_factor
        self.class_weights = initial_weights if initial_weights else {}
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        """Adjust weights based on validation performance"""
        # Get current validation loss
        current_val_loss = logs.get('val_loss')
        
        # If validation loss is improving, keep current weights
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            return
        
        # Otherwise, adjust weights for minority classes
        if epoch % 5 == 0:  # Adjust every 5 epochs
            y_pred = self.model.predict(self.X_train)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(self.y_train, axis=1)
            
            # Calculate class-wise accuracy
            class_accuracy = {}
            for cls in range(len(self.class_weights)):
                cls_indices = (y_true_classes == cls)
                if np.sum(cls_indices) > 0:
                    class_accuracy[cls] = np.mean(y_pred_classes[cls_indices] == cls)
                else:
                    class_accuracy[cls] = 0
            
            # Adjust weights - increase weight for classes with lower accuracy
            for cls in self.class_weights:
                if class_accuracy[cls] < np.mean(list(class_accuracy.values())):
                    self.class_weights[cls] *= (1 + self.adjustment_factor)
                else:
                    self.class_weights[cls] *= (1 - self.adjustment_factor * 0.5)
            
            print(f"\nEpoch {epoch}: Updated class weights: {self.class_weights}")


# %%

# 7. Enhanced Training Function
def train_model_with_advanced_techniques(model, X_train, y_train, X_val, y_val, epochs=150, batch_size=64, symbol=None):
    """Train model with enhanced class balancing techniques"""
    # Convert y_train to integer labels (from one-hot encoded)
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_labels = np.argmax(y_train, axis=1)
    else:
        y_train_labels = y_train.copy()
    
    # Compute class weights with more emphasis on minority classes
    class_counts = np.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    class_weights = {
        i: (total_samples / (len(np.unique(y_train_labels)) * count)) * 1.5 
        if i != 1 else (total_samples / (len(np.unique(y_train_labels)) * count))
        for i, count in enumerate(class_counts)
    }
    
    # Dynamic class weight callback
    dynamic_weights = DynamicClassWeightCallback(
        X_train, y_train, 
        initial_weights=class_weights,
        adjustment_factor=0.05
    )
    
    # Use symbol-specific checkpoint path if provided
    checkpoint_path = f"{symbol}_best_model.keras" if symbol else "best_model.keras"
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        dynamic_weights
    ]

    # Train with more epochs and larger batch size
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model


# %%

# 8. Advanced Trading Signal Generation
def generate_advanced_trading_signals(model, X_test, df_test, confidence_threshold=0.6):
    """Generate trading signals with confidence filtering and position sizing"""
    # Get predictions with probabilities
    y_pred_proba = model.predict(X_test)
    
    # Create signals DataFrame
    signals = pd.DataFrame(index=df_test.index[:len(y_pred_proba)])
    signals['pred_down_prob'] = y_pred_proba[:, 0]
    signals['pred_neutral_prob'] = y_pred_proba[:, 1]
    signals['pred_up_prob'] = y_pred_proba[:, 2]
    
    # Apply confidence threshold
    signals['predicted_class'] = np.argmax(y_pred_proba, axis=1)
    signals['confidence'] = np.max(y_pred_proba, axis=1)
    
    # Initialize position column
    signals['position'] = 0  # Default is no position
    
    # Apply confidence threshold for taking positions
    signals.loc[(signals['predicted_class'] == 2) & 
                (signals['confidence'] > confidence_threshold), 'position'] = 1  # Long
    
    signals.loc[(signals['predicted_class'] == 0) & 
                (signals['confidence'] > confidence_threshold), 'position'] = -1  # Short
    
    # Position sizing based on confidence (scale between 0.5 and 1.0)
    signals['position_size'] = signals['position'] * (signals['confidence'] - 0.5) * 2
    signals.loc[signals['position_size'] < 0, 'position_size'] = signals.loc[signals['position_size'] < 0, 'position_size'] * -1
    
    # Add price data
    signals['price'] = df_test['Close'].values[:len(signals)]
    
    # Add market volatility for risk adjustment
    if 'Volatility_20d' in df_test.columns:
        signals['volatility'] = df_test['Volatility_20d'].values[:len(signals)]
        # Adjust position size inversely to volatility (smaller positions in high volatility)
        signals['position_size'] = signals['position_size'] / (1 + signals['volatility'] * 10)
    
    # Calculate returns
    signals['market_return'] = np.log(signals['price'] / signals['price'].shift(1))
    signals['strategy_return'] = signals['position'].shift(1) * signals['market_return']
    signals['sized_strategy_return'] = signals['position_size'].shift(1) * signals['market_return']
    
    # Calculate drawdown
    signals['cumulative_market_return'] = np.exp(signals['market_return'].cumsum()) - 1
    signals['cumulative_strategy_return'] = np.exp(signals['sized_strategy_return'].cumsum()) - 1
    
    signals['drawdown'] = signals['cumulative_strategy_return'] - signals['cumulative_strategy_return'].cummax()
    
    # Calculate performance metrics
    total_return = np.exp(signals['sized_strategy_return'].sum()) - 1
    annual_return = np.exp(signals['sized_strategy_return'].mean() * 252) - 1
    sharpe_ratio = np.sqrt(252) * signals['sized_strategy_return'].mean() / signals['sized_strategy_return'].std()
    max_drawdown = signals['drawdown'].min()
    win_rate = len(signals[signals['sized_strategy_return'] > 0]) / len(signals[signals['sized_strategy_return'] != 0])
    
    # Calculate profit factor
    gross_profits = signals.loc[signals['sized_strategy_return'] > 0, 'sized_strategy_return'].sum()
    gross_losses = abs(signals.loc[signals['sized_strategy_return'] < 0, 'sized_strategy_return'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    performance_metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }
    
    return signals, performance_metrics

# 9. Enhanced Performance Visualization
def visualize_trading_performance(signals, performance_metrics, symbol_name):
    """Create comprehensive visualizations of trading performance"""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Cumulative returns comparison
    ax1 = fig.add_subplot(3, 1, 1)
    signals['cumulative_market_return'].plot(ax=ax1, label=f'{symbol_name} Return', color='blue', alpha=0.7)
    signals['cumulative_strategy_return'].plot(ax=ax1, label='Strategy Return', color='green')
    ax1.set_title(f'Cumulative Returns Comparison - {symbol_name}')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Drawdown
    ax2 = fig.add_subplot(3, 1, 2)
    signals['drawdown'].plot(ax=ax2, color='red')
    ax2.set_title('Strategy Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True)
    
    # Plot 3: Position and price
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(signals.index, signals['price'], color='black', alpha=0.7)
    
    # Add buy/sell markers
    buy_signals = signals[signals['position'].diff() > 0]
    sell_signals = signals[signals['position'].diff() < 0]
    
    ax3.scatter(buy_signals.index, buy_signals['price'], marker='^', color='green', s=100, label='Buy')
    ax3.scatter(sell_signals.index, sell_signals['price'], marker='v', color='red', s=100, label='Sell')
    
    ax3.set_title(f'Trading Signals - {symbol_name}')
    ax3.set_ylabel('Price')
    ax3.legend()
    
    # Add performance metrics as text
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
    plt.savefig(f"{symbol_name}_trading_performance.png")
    plt.show()


# %%
# 10. Main Execution Function for Single Symbol
def run_trading_system_for_symbol(data_path, symbol, confidence_threshold=0.6):
    """Run complete trading system pipeline for a single stock symbol"""
    # Load data
    print(f"Loading data for {symbol}...")
    df = pd.read_csv(data_path)
    
    # Filter data for the specific symbol
    if 'Symbol' in df.columns:
        df = df[df['Symbol'] == symbol].copy()
        print(f"Filtered data for {symbol}. Count of rows: {len(df)}")
    else:
        print(f"No Symbol column found. Assuming data is already for {symbol}.")
    
    if len(df) == 0:
        print(f"No data found for symbol {symbol}.")
        return None, None, None
    
    # Convert Date column to datetime if it exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Preprocess data
    print("Preprocessing data...")
    df = preprocess_stock_data(df)
    print(f"Missing values after imputation: {df.isna().sum().sum()}")
    
    # Define feature categories
    price_indicators = ['Close', 'Returns', 'Log_Returns', 'Price_Range', 'Price_Range_Pct']
    moving_averages = [col for col in df.columns if col.startswith('MA_') or 
                        col.startswith('EMA_') or col.startswith('Returns_')]
    volatility_metrics = [col for col in df.columns if col.startswith('Volatility_') or 
                            col.startswith('Volume_MA_') or col.startswith('BB_Width_')]
    technical_indicators = ['RSI_9', 'RSI_14', 'RSI_25', 'MACD', 'Signal_Line', 'MACD_Histogram',
                            'Momentum_14', 'ROC_14', 'MFI_14', 'MFI_28'] + \
                            [col for col in df.columns if col.startswith('Channel_Width_')]
    volume_indicators = ['OBV', 'Volume_Ratio', 'Volume_StdDev']
    fundamental_features = ['PE_Ratio', 'PB_Ratio', 'Dividend_Yield', 'Profit_Margin', 
                            'Beta', 'Enterprise_Value', 'Forward_EPS', 'Trailing_EPS']
    market_features = ['Market_Return', 'Market_Volatility', 'Rolling_Beta', 'VIX', 'VIX_MA_10']
    
    # Combine all features
    all_features = price_indicators + moving_averages + volatility_metrics + \
                    technical_indicators + volume_indicators + \
                    fundamental_features + market_features
    
    # Remove any features not in the dataframe
    features = [f for f in all_features if f in df.columns]
    print(f"Using {len(features)} features")
    
    # Feature selection with mutual information
    selected_features = select_features_with_mi(df, features, 'target', n_select=30)
    print(f"Selected {len(selected_features)} features")
    
    # Create correlation tensor features
    tensor_indicators = [ind for ind in [
        'RSI_14', 'MACD', 'Momentum_14', 'ROC_14', 'MFI_14', 
        'Volume_Ratio', 'Returns', 'Volatility_20d'
    ] if ind in df.columns]
    
    # Prepare sequence data
    print("Preparing sequence data...")
    sequence_length = 30  # 30 days of historical data
    X, y = prepare_sequence_data(df, sequence_length, 'target', selected_features)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Convert target to one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)
    
    # Split data using time series split
    print("Splitting data...")
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train, y_train = X[:train_size], y_onehot[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y_onehot[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y_onehot[train_size+val_size:]
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Testing set: {X_test.shape}, {y_test.shape}")
    
    # Check class distribution
    class_counts = np.sum(y_train, axis=0)
    print(f"Class distribution in training set:")
    print(f"Down: {class_counts[0]}, Neutral: {class_counts[1]}, Up: {class_counts[2]}")
    
    # Check if model exists
    model_path = f"{symbol}_model.keras"
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = tf.keras.models.load_model(model_path)
    else:
        print(f"Model not found. Building and training a new model...")
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_enhanced_cnn_bilstm_model(input_shape, num_classes=3)
        model.summary()
        
        # Train model
        print("Training model...")
        history, model = train_model_with_advanced_techniques(
            model, X_train, y_train, X_val, y_val, 
            epochs=100, batch_size=32, symbol=symbol
        )
        
        # Save the trained model
        print(f"Saving trained model to {model_path}")
        model.save(model_path)
        np.save(f"{symbol}_X_test.npy", X_test)
        np.save(f"{symbol}_y_test.npy", y_test)
    
    # Evaluate model performance
    print("Evaluating model...")
    # Convert one-hot encoded targets back to class indices
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Get model predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision = precision_score(y_test_labels, y_pred, average='weighted')
    recall = recall_score(y_test_labels, y_pred, average='weighted')
    f1 = f1_score(y_test_labels, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Neutral', 'Up'],
                yticklabels=['Down', 'Neutral', 'Up'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {symbol}')
    plt.savefig(f"{symbol}_confusion_matrix.png")
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {symbol}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {symbol}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{symbol}_training_history.png")
    plt.show()
    
    # Generate trading signals
    print("Generating trading signals...")
    df_test = df.iloc[train_size+val_size+sequence_length:]
    signals, performance_metrics = generate_advanced_trading_signals(model, X_test, df_test, confidence_threshold)
    
    # Visualize trading performance
    visualize_trading_performance(signals, performance_metrics, symbol)
    
    # Save model and data
    
    signals.to_csv(f"{symbol}_signals.csv")
    
    return model, signals, performance_metrics


# %%
# Enhanced feature importance analysis
def analyze_feature_importance_for_symbol(model, X_test, y_test, feature_names, symbol):
    """Analyze feature importance using permutation importance"""
    from sklearn.inspection import permutation_importance
    
    # Convert model predictions to class labels
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Calculate baseline accuracy
    baseline_accuracy = accuracy_score(y_test_classes, y_pred_classes)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Reshape X_test for sklearn compatibility
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
    
    # Calculate permutation importance
    result = permutation_importance(
        lambda x: np.argmax(model.predict(x.reshape(-1, X_test.shape[1], X_test.shape[2])), axis=1),
        X_test_reshaped,
        y_test_classes,
        n_repeats=10,
        random_state=42,
        scoring='accuracy'
    )
    
    # Create DataFrame with importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20])
    plt.xlabel('Permutation Importance (decrease in accuracy)')
    plt.title(f'Feature Importance for {symbol} (Top 20)')
    plt.tight_layout()
    plt.savefig(f"{symbol}_feature_importance.png")
    plt.show()
    
    return importance_df
def detect_market_regime(df, window=60):
    """Detect market regime based on volatility and trend"""
    # Calculate rolling volatility
    df['volatility'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
    
    # Calculate rolling trend (slope of linear regression)
    def rolling_slope(series, window):
        slopes = []
        for i in range(len(series) - window + 1):
            x = np.arange(window)
            y = series.iloc[i:i+window].values
            slope, _, _, _, _ = linregress(x, y)
            slopes.append(slope)
        return pd.Series(np.nan, index=series.index).iloc[window-1:].combine_first(pd.Series(slopes, index=series.index[window-1:len(slopes)+window-1]))
    
    df['trend'] = rolling_slope(df['Close'], window)
    
    # Classify regimes
    # High volatility, positive trend = 'Bull Volatile'
    # High volatility, negative trend = 'Bear Volatile'
    # Low volatility, positive trend = 'Bull Stable'
    # Low volatility, negative trend = 'Bear Stable'
    
    volatility_threshold = df['volatility'].mean() + 0.5 * df['volatility'].std()
    
    df['regime'] = np.where(
        df['volatility'] > volatility_threshold,
        np.where(df['trend'] > 0, 'Bull Volatile', 'Bear Volatile'),
        np.where(df['trend'] > 0, 'Bull Stable', 'Bear Stable')
    )
    
    return df
def evaluate_signal_quality(signals, prediction_horizon=5):
    """Evaluate signal quality using future price movements"""
    # Calculate future returns for different horizons
    for days in range(1, prediction_horizon + 1):
        signals[f'future_return_{days}d'] = signals['price'].pct_change(periods=days).shift(-days)
    
    # Evaluate buy signals
    buy_signals = signals[signals['position'] == 1]
    if len(buy_signals) > 0:
        buy_accuracy = {}
        for days in range(1, prediction_horizon + 1):
            buy_accuracy[days] = np.mean(buy_signals[f'future_return_{days}d'] > 0)
        
        print("Buy Signal Quality:")
        for days, acc in buy_accuracy.items():
            print(f"  {days}-day accuracy: {acc:.4f}")
    
    # Evaluate sell signals
    sell_signals = signals[signals['position'] == -1]
    if len(sell_signals) > 0:
        sell_accuracy = {}
        for days in range(1, prediction_horizon + 1):
            sell_accuracy[days] = np.mean(sell_signals[f'future_return_{days}d'] < 0)
        
        print("Sell Signal Quality:")
        for days, acc in sell_accuracy.items():
            print(f"  {days}-day accuracy: {acc:.4f}")
    
    # Calculate signal timeliness (how early signals are generated)
    # For buy signals: how many days before a significant uptrend
    # For sell signals: how many days before a significant downtrend
    
    # Define significant moves (e.g., 2% in 5 days)
    threshold = 0.02
    
    signals['significant_up'] = np.max([signals[f'future_return_{days}d'] > threshold 
                                        for days in range(1, prediction_horizon + 1)], axis=0)
    signals['significant_down'] = np.max([signals[f'future_return_{days}d'] < -threshold 
                                          for days in range(1, prediction_horizon + 1)], axis=0)
    
    # Return signal quality metrics
    signal_metrics = {
        'buy_precision': buy_accuracy[1] if len(buy_signals) > 0 else 0,
        'sell_precision': sell_accuracy[1] if len(sell_signals) > 0 else 0,
        'buy_signals_count': len(buy_signals),
        'sell_signals_count': len(sell_signals),
        'buy_win_rate': np.mean(buy_signals['strategy_return'] > 0) if len(buy_signals) > 0 else 0,
        'sell_win_rate': np.mean(sell_signals['strategy_return'] > 0) if len(sell_signals) > 0 else 0
    }
    
    return signal_metrics
def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Tune hyperparameters using grid search"""
    param_grid = {
        'filters': [[64, 128, 256], [128, 256, 512]],
        'lstm_units': [[64, 32], [128, 64]],
        'dropout_rates': [[0.3, 0.4, 0.5], [0.2, 0.3, 0.4]],
        'learning_rates': [0.001, 0.0005, 0.0001]
    }
    
    best_val_accuracy = 0
    best_params = None
    results = []
    
    # Simple grid search implementation
    for filters in param_grid['filters']:
        for lstm_units in param_grid['lstm_units']:
            for dropout_rates in param_grid['dropout_rates']:
                for lr in param_grid['learning_rates']:
                    print(f"Testing: filters={filters}, lstm_units={lstm_units}, "
                          f"dropout={dropout_rates}, lr={lr}")
                    
                    # Build model with current parameters
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    model = build_model_with_params(
                        input_shape, 
                        filters=filters,
                        lstm_units=lstm_units,
                        dropout_rates=dropout_rates,
                        learning_rate=lr
                    )
                    
                    # Train with early stopping
                    early_stopping = EarlyStopping(
                        monitor='val_accuracy', 
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,  # Reduced epochs for hyperparameter search
                        batch_size=32,
                        callbacks=[early_stopping],
                        verbose=0
                    )
                    
                    # Get best validation accuracy
                    val_accuracy = max(history.history['val_accuracy'])
                    print(f"Validation accuracy: {val_accuracy:.4f}")
                    
                    # Track results
                    results.append({
                        'filters': filters,
                        'lstm_units': lstm_units,
                        'dropout_rates': dropout_rates,
                        'learning_rate': lr,
                        'val_accuracy': val_accuracy
                    })
                    
                    # Update best parameters
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        best_params = {
                            'filters': filters,
                            'lstm_units': lstm_units,
                            'dropout_rates': dropout_rates,
                            'learning_rate': lr
                        }
    
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    
    return best_params, pd.DataFrame(results)
def optimize_decision_thresholds(model, X_val, y_val, df_val):
    """Optimize decision thresholds for trading signals"""
    # Get predictions
    y_pred_proba = model.predict(X_val)
    
    # Initialize variables
    best_sharpe = 0
    best_threshold_up = 0.5
    best_threshold_down = 0.5
    
    # Grid search over thresholds
    for threshold_up in np.arange(0.5, 0.95, 0.05):
        for threshold_down in np.arange(0.5, 0.95, 0.05):
            # Generate signals with current thresholds
            signals = pd.DataFrame(index=range(len(y_pred_proba)))
            signals['pred_down_prob'] = y_pred_proba[:, 0]
            signals['pred_neutral_prob'] = y_pred_proba[:, 1]
            signals['pred_up_prob'] = y_pred_proba[:, 2]
            
            # Apply thresholds
            signals['position'] = 0  # Default is no position
            signals.loc[signals['pred_up_prob'] > threshold_up, 'position'] = 1  # Long
            signals.loc[signals['pred_down_prob'] > threshold_down, 'position'] = -1  # Short
            
            # Add price data
            if len(df_val) >= len(signals):
                signals['price'] = df_val['Close'].values[:len(signals)]
                
                # Calculate returns
                signals['market_return'] = np.log(signals['price'] / signals['price'].shift(1))
                signals['strategy_return'] = signals['position'].shift(1) * signals['market_return']
                
                # Calculate Sharpe ratio
                sharpe_ratio = np.sqrt(252) * signals['strategy_return'].mean() / signals['strategy_return'].std() if signals['strategy_return'].std() > 0 else 0
                
                # Update best thresholds
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    best_threshold_up = threshold_up
                    best_threshold_down = threshold_down
    
    print(f"Best thresholds: Up={best_threshold_up:.2f}, Down={best_threshold_down:.2f}")
    print(f"Best Sharpe ratio: {best_sharpe:.4f}")
    
    return best_threshold_up, best_threshold_down

# %%
# Run the entire pipeline
if __name__ == "__main__":
    model, signals, performance = run_trading_system_for_symbol('sp500_master_data.csv','TSLA',confidence_threshold=0.65)
    
    print("\nTrading Performance Metrics:")
    print(f"Total Return: {performance['total_return']*100:.2f}%")
    print(f"Annual Return: {performance['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {performance['max_drawdown']*100:.2f}%")
    print(f"Win Rate: {performance['win_rate']*100:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")


