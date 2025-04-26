# CNN-BiLSTM Trading System - Technical Documentation

## Background and Motivation

This document provides a comprehensive explanation of the CNN-BiLSTM trading system implemented in Final.py. The system uses deep learning to predict stock price movements and generate trading signals based on those predictions.

## Key Challenges and Analysis

1. **Time Series Analysis**: Stock data is time-dependent, requiring specialized techniques for preprocessing and modeling.
2. **Feature Engineering**: Converting raw price data into meaningful technical indicators.
3. **Class Imbalance**: Financial datasets often have imbalanced buy/sell signals.
4. **Model Architecture Design**: Creating an effective neural network that captures both spatial (CNN) and temporal (BiLSTM) patterns.
5. **Trading Signal Generation**: Converting model predictions into actionable trading decisions.
6. **Performance Evaluation**: Measuring both predictive and trading performance.

## High-level Task Breakdown

1. **Data Preparation**
   - Load stock data
   - Calculate technical indicators
   - Create sequences for time series modeling
   - Split data into training, validation, and test sets
   - Apply SMOTE for class balancing

2. **Model Architecture**
   - Convolutional layers for feature extraction
   - Bidirectional LSTM layers for capturing temporal dependencies
   - Attention mechanism for focusing on important time steps
   - Dense layers for final prediction

3. **Training Process**
   - Train the model using resampled data
   - Validate on a separate validation set
   - Monitor accuracy and loss metrics

4. **Signal Generation**
   - Convert model predictions to trading signals
   - Apply confidence thresholds for decision making
   - Generate positions (-1 for sell, 0 for hold, 1 for buy)

5. **Performance Evaluation**
   - Calculate predictive metrics (accuracy, precision, recall, F1)
   - Calculate trading metrics (returns, Sharpe ratio, drawdown)
   - Visualize results through various plots

6. **Multi-Symbol Analysis**
   - Process multiple stock symbols
   - Compare performance across symbols
   - Generate aggregate statistics

## Detailed Data Flow

### 1. Data Loading and Preprocessing

![Data Preprocessing Flow](https://i.imgur.com/GWOODg9.png)

The system begins by loading stock data from a CSV file and filtering for a specific symbol. The data preprocessing pipeline includes:

1. **Technical Indicator Calculation**:
   - Moving Averages (MA_50, MA_200)
   - Relative Strength Index (RSI)
   - Moving Average Convergence Divergence (MACD)
   - Bollinger Bands

2. **Sequence Creation**:
   - For each time step t, create a sequence of the previous `seq_length` (30) days
   - Each sequence includes multiple features (price, volume, technical indicators)
   - Target variable: 1 if next day's return > 0, else 0 (binary classification)

3. **Feature Selection**:
   - Use mutual information to identify the most informative features
   - Standardize features using MinMaxScaler

4. **Data Splitting**:
   - Training set: 70% of data
   - Validation set: 15% of data
   - Test set: 15% of data

5. **Class Balancing**:
   - Apply SMOTE to the training set to address class imbalance

### 2. Model Architecture

![CNN-BiLSTM Architecture](https://i.imgur.com/bQJZoGL.png)

The model uses a hybrid CNN-BiLSTM architecture with attention mechanism:

1. **Convolutional Layers**:
   - Input shape: (sequence_length, num_features)
   - First Conv1D layer: 64 filters, kernel size 3
   - MaxPooling and Dropout for regularization

2. **Bidirectional LSTM Layers**:
   - First BiLSTM: 128 units, returns sequences
   - Second BiLSTM: 32 units, returns sequences
   - Final BiLSTM: 32 units, returns final state

3. **Attention Mechanism**:
   - Calculates attention weights for each time step
   - Weights the importance of different time steps in the sequence

4. **Dense Layers**:
   - 32 units with ReLU activation
   - Final output layer with sigmoid activation for binary classification

### 3. Training Process

The model is trained using:
- Optimizer: Adam with learning rate 0.001
- Loss function: Binary cross-entropy
- Batch size: 32
- Epochs: 50
- Early stopping based on validation loss

### 4. Signal Generation

![Trading Signal Generation](https://i.imgur.com/JfA9XKV.png)

The system generates trading signals from model predictions:

1. **Confidence Calculation**:
   - Probability output from the model (0 to 1)
   - Apply confidence threshold (default: 0.6)

2. **Position Generation**:
   - Long (1): When predicted class is 1 and confidence > threshold
   - Short (-1): When predicted class is 0 and confidence > threshold
   - Neutral (0): When confidence ≤ threshold

3. **Position Sizing**:
   - Scale position size based on prediction confidence
   - Higher confidence = larger position size

### 5. Performance Evaluation

The system evaluates performance using multiple metrics:

1. **Predictive Performance**:
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix

2. **Trading Performance**:
   - Total Return
   - Annual Return
   - Sharpe Ratio
   - Maximum Drawdown
   - Win Rate
   - Profit Factor

3. **Trade Simulation**:
   - Simulates trades with stop-loss and take-profit
   - Tracks portfolio value over time

### 6. Visualization

The system generates various visualizations:

1. **Model Performance**:
   - Training/validation accuracy and loss curves
   - Confusion matrix

2. **Trading Performance**:
   - Cumulative returns comparison
   - Drawdown chart
   - Buy/sell signals on price chart

3. **Multi-Symbol Analysis**:
   - Performance metrics heatmap
   - Top performers by return
   - Risk-return scatter plot
   - Trading dashboard with multiple metrics

## Project Status Board

- [x] Understand data preprocessing pipeline
- [x] Examine model architecture details
- [x] Review training process and hyperparameters
- [x] Analyze signal generation logic
- [x] Evaluate performance metrics calculation
- [x] Study visualization components
- [x] Compare multi-symbol analysis approach
- [~] Implement class weighting in model training to address class imbalance (in progress)
- [~] Enhance feature selection using mutual information for each symbol (in progress)
- [~] Add more advanced regularization (e.g., early stopping, learning rate reduction) (in progress)
- [ ] Re-run training and evaluate improvements in precision and F1

## Executor's Feedback or Assistance Requests

- Implementation of all planned improvements has started: class weighting, dynamic feature selection, and advanced regularization.
- Will update after each subtask is completed and before re-running the full training.

## Lessons

*This section will document learnings, challenges, and solutions encountered during the project.* 