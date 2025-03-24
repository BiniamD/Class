import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# Load your preprocessed data
# Replace 'your_preprocessed_data.csv' with your actual file path
df = pd.read_csv('sp500_master_data.csv')

# If necessary, convert date column and set as index
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

# Create the target variable if it doesn't exist
price_change_threshold = 0.005
if 'target' not in df.columns:
    df['target'] = np.where(df['Returns'].shift(-1) > price_change_threshold, 2,
                          np.where(df['Returns'].shift(-1) < -price_change_threshold, 0, 1))
    df = df.dropna(subset=['target'])
    df['target'] = df['target'].astype(int)

# 1. Dataset Size and Shape
print("Dataset Information:")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Number of trading days: {df.shape[0]}")

# 2. Class Distribution
print("\nClass Distribution:")
class_counts = df['target'].value_counts()
print(class_counts)
print(f"Class 0 (Down): {class_counts.get(0, 0)} ({class_counts.get(0, 0) / len(df) * 100:.2f}%)")
print(f"Class 1 (Neutral): {class_counts.get(1, 0)} ({class_counts.get(1, 0) / len(df) * 100:.2f}%)")
print(f"Class 2 (Up): {class_counts.get(2, 0)} ({class_counts.get(2, 0) / len(df) * 100:.2f}%)")

# Visualize class distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='target', data=df)
plt.title('Class Distribution')
plt.xlabel('Target Class (0: Down, 1: Neutral, 2: Up)')
plt.ylabel('Count')
#plt.savefig('class_distribution.png')
plt.show()
plt.close()

# 3. Feature Analysis
# Filter out non-numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numeric_cols:
    numeric_cols.remove('target')

# Calculate correlation with target
print("\nTop 15 Features by Correlation with Target:")
correlations = df[numeric_cols].corrwith(df['target']).abs().sort_values(ascending=False)
print(correlations.head(15))

# Visualize correlations
plt.figure(figsize=(12, 8))
top_features = correlations.head(15).index
correlation_matrix = df[list(top_features) + ['target']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Top Features')
plt.tight_layout()
#plt.savefig('correlation_matrix.png')
plt.show()
plt.close()

# 4. Feature Importance using Mutual Information (better for non-linear relationships)
try:
    print("\nTop 15 Features by Mutual Information with Target:")
    X = df[numeric_cols].fillna(0)
    y = df['target']
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': numeric_cols, 'MI Score': mi_scores})
    mi_df = mi_df.sort_values('MI Score', ascending=False)
    print(mi_df.head(15))
    
    # Visualize mutual information
    plt.figure(figsize=(12, 8))
    sns.barplot(x='MI Score', y='Feature', data=mi_df.head(15))
    plt.title('Feature Importance by Mutual Information')
    plt.tight_layout()
    #plt.savefig('mutual_information.png')
    plt.show()
    plt.close()
except Exception as e:
    print(f"Error calculating mutual information: {e}")

# 5. Time Series Characteristics
print("\nTime Series Characteristics:")
# Calculate returns volatility
if 'Returns' in df.columns:
    rolling_std = df['Returns'].rolling(window=30).std()
    print(f"Average 30-day volatility: {rolling_std.mean():.4f}")
    print(f"Min 30-day volatility: {rolling_std.min():.4f}")
    print(f"Max 30-day volatility: {rolling_std.max():.4f}")
    
    # Plot volatility over time
    plt.figure(figsize=(14, 7))
    rolling_std.plot()
    plt.title('30-Day Rolling Volatility')
    plt.ylabel('Standard Deviation of Returns')
    #plt.savefig('volatility.png')
    plt.show()
    plt.close()

# 6. Feature Distribution Analysis
print("\nFeature Distribution Analysis:")
# Select top 5 features by correlation
top_5_features = correlations.head(5).index.tolist()
for feature in top_5_features:
    if feature in df.columns:
        print(f"\n{feature} statistics:")
        print(df[feature].describe())
        
        # Plot distribution
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='target', y=feature, data=df)
        plt.title(f'{feature} by Target Class')
        
        plt.tight_layout()
        #plt.savefig(f'{feature}_analysis.png')
        plt.show()
        plt.close()

# 7. Check for missing values
print("\nMissing Values Analysis:")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if len(missing_values) > 0:
    print(missing_values)
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Percentage missing: {df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
else:
    print("No missing values found.")

print("\nAnalysis complete. Image files have been saved.")