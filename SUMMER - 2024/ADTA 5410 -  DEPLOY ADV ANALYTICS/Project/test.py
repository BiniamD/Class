# %% [markdown]
# What Factors Affect STEM Degree Completion Rates at Public Universities in the United States?

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# Load the datasets
completions = pd.read_csv('IPEDS_Completions.csv')
Directory = pd.read_csv('IPEDS_Directory.csv')
Chareges = pd.read_csv('IPEDS_Charges.csv')

# %%
# shape of the loaded Datasets withe lable  as a table
print('completions shape:', completions.shape)
print('Directory shape:', Directory.shape)
print('Chareges shape:', Chareges.shape)

# %%
completions.head()

# %%
Directory.head(2)

# %%
Chareges.head()

# %% [markdown]
# ## Clean Data and simplify columns 

# %%
# Filter for public universities
public_universities =  Directory[Directory['CONTROL'] == 1]
public_university_ids = public_universities['UNITID'].unique()

# %%

# Define the CIP codes for STEM fields
# Replace with actual CIP codes for STEM fields
stem_cip_codes = [11, 14, 15, 26, 27, 40, 41, 52, 54, 62, 91, 15, 26, 52, 54, 62, 91]

# %%
# Filter completions for public universities and STEM fields CIP codes
completions['CIPCODE'] = completions['CIPCODE'].astype(str).str.split('.').str[0].astype(int)

# %%
# Filter for public universities and STEM programs
completions_stem = completions[(completions['UNITID'].isin(public_university_ids)) & 
                               (completions['CIPCODE'].isin(stem_cip_codes))]                               
#remove all the columns that start with XC
completions_stem = completions_stem.loc[:,~completions_stem.columns.str.startswith('XC')]
# remove all the columns that end with T 
completions_stem = completions_stem.loc[:,~completions_stem.columns.str.endswith('T')]
# remove CIPCODE , MAJORNUM , AWLEVEL 
completions_stem = completions_stem.drop(['CIPCODE', 'MAJORNUM', 'AWLEVEL'], axis=1)
#make unitid as categorical
completions_stem['UNITID'] = completions_stem['UNITID'].astype('category')
# aggregate the data by UNITID
completions_stem = completions_stem.groupby('UNITID').sum().reset_index()

# %%

#remove all the columns that start with XC
completions_stem = completions_stem.loc[:,~completions_stem.columns.str.startswith('XC')]
# remove all the columns that end with T 
completions_stem = completions_stem.loc[:,~completions_stem.columns.str.endswith('T')]
# remove CIPCODE , MAJORNUM , AWLEVEL 
completions_stem = completions_stem.drop(['CIPCODE', 'MAJORNUM', 'AWLEVEL'], axis=1)
#make unitid as categorical
completions_stem['UNITID'] = completions_stem['UNITID'].astype('category')
# aggregate the data by UNITID
completions_stem = completions_stem.groupby('UNITID').sum().reset_index()

# %%
# add CTOTALM and CTOTALW add it to the end 
completions_stem['CTOTAL'] = completions_stem['CTOTALM'] + completions_stem['CTOTALW']

# %%
completions_stem.describe()

# %%
#filter only UNITID TUITION1 FEE1 HRCHG1
Chareges = Chareges[['UNITID', 'TUITION1', 'FEE1', 'HRCHG1']]


# %%
# merge completions_stem and Chareges
data = pd.merge(completions_stem, Chareges, on='UNITID', how='inner')

# %%
# Descriptive statistics for the target variable CTOTAL 
descriptive_stats = data['CTOTAL'].describe()

# Display descriptive statistics
print("Descriptive Statistics for the Target Variable 'CTOTAL':")
print(descriptive_stats)

# %%
# Plot the distribution of the target variable CTOTAL
plt.figure(figsize=(8, 6))
sns.histplot(data['CTOTAL'], bins=30, kde=True, color='blue')
plt.title('Distribution of CTOTAL')
plt.xlabel('CTOTAL')
plt.ylabel('Frequency')
plt.show()


# %%
# visualize the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# %%
#visualize the scatter plot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='TUITION1', y='CTOTAL', data=data, color='blue')
plt.title('CTOTAL vs. TUITION1')
plt.xlabel('TUITION1')
plt.ylabel('CTOTAL')
plt.show()

# %%
# target is CTOTAL
target  = data['CTOTAL']
# features are all the columns except CTOTAL
features  = data.drop('CTOTAL', axis=1)

# %%
# Preprocess numerical features
numeric_features = features.select_dtypes(include=['float64', 'int64'])
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Preprocess categorical features
categorical_features = features.select_dtypes(include=['category'])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# %%
# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features.columns),
        ('cat', categorical_transformer, categorical_features.columns)])

# %%
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Define and train the model using a pipeline
model = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
model.fit(X_train, y_train)

# %%
# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# %%
#  Print the evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')




# %%

# Plot actual vs predicted completions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Completions')
plt.ylabel('Predicted Completions')
plt.title('Actual vs Predicted STEM Completions')
plt.show()

# %%
import numpy as np

# Extract and display coefficients
coefficients = model.named_steps['regressor'].coef_
feature_names = numeric_features

feature_importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
print(feature_importance.sort_values(by='Coefficient', ascending=False))

# %%
# Ensure that feature names are strings
feature_importance['Feature'] = feature_importance['Feature'].apply(lambda x: ''.join(x) if isinstance(x, tuple) else x)

# Sort feature importance by coefficient value
feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

# Plot the feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
plt.xlabel('Coefficient')
plt.title('Feature Importance for STEM Degree Completion Rates')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest coefficient on top
plt.show()


# %%

# Plot the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Completions')
plt.ylabel('Predicted Completions')
plt.title('Actual vs Predicted Completions')
plt.show()


# %%
# Mean Squared Error: 1.1176410341073307e-23
#R-squared: 1.0


