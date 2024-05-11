
#This code installs the `ucimlrepo` package.
#pip install ucimlrepo

import pandas as pd
import numpy as np
import dmba
import seaborn as sns
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.decomposition import PCA

# filter warnings
import warnings
warnings.filterwarnings("ignore")


import dmba
from dmba import classificationSummary, gainsChart, liftChart
from dmba.metric import AIC_score



from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  



# variable information 
#breast_cancer_wisconsin_diagnostic.data
#create dataframe using both x and y
df = pd.concat([X, y], axis=1)

#save the data to a csv file
#df.to_csv('breast_cancer_wisconsin_diagnostic.csv', index=False)

# ### Basic information about the dataset
# 
print("First 5 rows of the dataset:")
df.head()

df.shape

print("\nDataset summary:")
df.describe()



print("\nMissing values in the dataset:")
df.isnull().sum()


#print the information about the dataset 
print(df.info())

 [markdown]
# #### Data Preparation
# 


#replace the target values with 0 and 1
#y = np.where(y == 'M', 1, 0)
#df['diagnosis'] = y

# ### EDA

# #### Distribution Analysis:


# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure to hold the subplots
plt.figure(figsize=(20, 15))

# Plot histograms for each numeric feature to understand distributions
for index, column in enumerate(df.columns[:-1], 1):  # Exclude the 'diagnosis' column
    plt.subplot(6, 5, index)
    sns.histplot(df[column], kde=True, element='step', color='blue')
    plt.title(column)

plt.tight_layout()
plt.show()

#HeatMap Correlations between the features
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True, fmt='.1f')
plt.show()

# Count plot for the 'diagnosis' to see class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Diagnosis', data=df)
plt.title('Class Distribution')
plt.show()

model = LogisticRegression()
model.fit(X, y)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
feature_importance.plot(kind='barh')
plt.show()

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Dimensionality Reduction using PCA
pca = PCA(n_components=2)  # Reduce dimensions to 2 for visualization or further analysis
X_pca = pca.fit_transform(X_scaled)


# Explained variance ratio for PCA components
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio for PCA Components:", explained_variance)





# Plotting the first two principal components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y['Diagnosis'].map({'M': 'Malignant', 'B': 'Benign'}))
plt.title('PCA of Breast Cancer Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(title='Diagnosis')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)


# Initializing and training the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


#Predicting on the test set
y_pred = log_reg.predict(X_test)

# Calculating performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label= 'M')
recall = recall_score(y_test, y_pred, pos_label= 'M')
f1 = f1_score(y_test, y_pred,pos_label= 'M')
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizing the confusion matrix
plt.figure(figsize=(7, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False,
            xticklabels=['Predicted Benign', 'Predicted Malignant'],
            yticklabels=['Actual Benign', 'Actual Malignant'])
plt.title('Confusion Matrix of Logistic Regression Model')
plt.show()


# Visualization of Performance Metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
barplot = sns.barplot(x=metrics, y=values, palette='viridis')
plt.ylim(0.9, 1)
plt.title('Enhanced Performance Metrics of Logistic Regression Model')
plt.ylabel('Score')
for p in barplot.patches:
    barplot.annotate(format(p.get_height(), '.4f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', 
                     size=10, xytext=(0, 8),
                     textcoords='offset points')
plt.show()



