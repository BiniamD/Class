import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv('breast_cancer_wisconsin.csv')

# Q1: Distribution of benign and malignant diagnoses
def distribution_diagnosis(df):
    print(df['diagnosis'].value_counts())

# Q2: Correlations between the features
def feature_correlations(df):
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f')
    plt.show()

# Q3: Impact of each feature on the diagnosis
def feature_impact(df):
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    model = LogisticRegression()
    model.fit(X, y)
    feature_importance = pd.Series(model.coef_[0], index=X.columns)
    feature_importance.plot(kind='barh')
    plt.show()

# Q4: Feature values difference between benign and malignant tumors
def feature_values_difference(df):
    benign = df[df['diagnosis'] == 'B']
    malignant = df[df['diagnosis'] == 'M']
    for column in df.columns[2:]:
        sns.boxplot(x='diagnosis', y=column, data=df)
        plt.show()

# Q5: Identifying clusters in the data
def identify_clusters(df):
    from sklearn.cluster import KMeans
    X = df.drop(['id', 'diagnosis'], axis=1)
    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(X)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
    plt.show()

# Q6: Predicting the diagnosis using machine learning models
def predict_diagnosis(df):
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Q7: Best machine learning model for the dataset
def best_model(df):
    # This is a placeholder for comparing different models, such as SVM, Random Forest, etc.
    pass

# Q8: Improving the accuracy of the predictions
def improve_accuracy(df):
    # This is a placeholder for techniques like feature selection, parameter tuning, etc.
    pass

# Q9: Patterns in misclassified cases
def misclassified_cases(df):
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    misclassified = np.where(y_test != y_pred)
    print(df.iloc[misclassified])

# Q10: Model performance on unseen data
def model_performance_unseen_data(df):
    # This is a placeholder for evaluating the model on a separate test set.
    pass

# Example usage
distribution_diagnosis(df)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('breast_cancer_wisconsin.csv')

# Basic information about the dataset
def basic_info(df):
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset summary:")
    print(df.describe())
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

# Distribution of the target variable
def target_distribution(df):
    sns.countplot(x='diagnosis', data=df)
    plt.title('Distribution of Diagnosis (M = Malignant, B = Benign)')
    plt.show()

# Distribution of each feature
def feature_distribution(df):
    features = df.columns[2:]
    for feature in features:
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

# Pairwise relationships between features
def pairwise_relationships(df):
    sns.pairplot(df.iloc[:, 1:6], hue='diagnosis')
    plt.show()

# Correlation matrix
def correlation_matrix(df):
    corr = df.iloc[:, 1:].corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

# Boxplots for each feature by diagnosis
def feature_boxplots(df):
    features = df.columns[2:]
    for feature in features:
        sns.boxplot(x='diagnosis', y=feature, data=df)
        plt.title(f'{feature} by Diagnosis')
        plt.show()

# Example usage
basic_info(df)
target_distribution(df)
feature_distribution(df)
pairwise_relationships(df)
correlation_matrix(df)
feature_boxplots(df)
