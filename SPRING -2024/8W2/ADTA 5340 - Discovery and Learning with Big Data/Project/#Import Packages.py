#Import Packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

# import the necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report

#Remove warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

base = GaussianNB()
ada_clf = AdaBoostClassifier(base_estimator=base, n_estimators=50, learning_rate=1, random_state=123)


#Load the Dataset ***Note: You must change the path
Homes = pd.read_csv("RDC_Inventory_Core_Metrics_County_History 2.csv")


#Preview Dataset in Python
display(Homes.head())

display(Homes.tail())

#Check the shape of the dataset
Homes.shape


#Categorical Variables Clean: Split County Name and State Column
Homes[['county', 'state']] = Homes['county_name'].str.split(',', expand=True)


#Categorical Variables Clean: Split Year and Month
Homes['month_date_yyyymm'] = pd.Series(Homes['month_date_yyyymm'], dtype="string")
Homes['year'] = Homes['month_date_yyyymm'].str[0:4]
Homes['month'] = Homes['month_date_yyyymm'].str[-2:]
display(Homes.head())


#For the purpose of the proect we only need some of the columns.   We can select them and rename the data.
HomesCleaned = Homes[['year', 'month', 'county', 'state', 'median_listing_price', 'active_listing_count', 'median_days_on_market', 'new_listing_count', 
               'price_increased_count', 'price_reduced_count', 'pending_listing_count', 'median_square_feet', 'average_listing_price',
               'total_listing_count']] 

display(HomesCleaned.head())
display(HomesCleaned.tail())


#Add new listing Prop
Homes['new_listing_Prop'] = Homes['new_listing_count'] / Homes['total_listing_count']
Homes.new_listing_Prop.head()

#Add price reduced Prop
Homes['price_reduced_Prop'] = Homes['price_reduced_count'] / Homes['total_listing_count']
Homes.price_reduced_Prop.head()

#Add price increased Prop
Homes['price_increased_Prop'] = Homes['price_increased_count'] / Homes['total_listing_count']
Homes.price_increased_Prop.tail()

#Add pending_listing_Prop
Homes['pending_listing_Prop'] = Homes['pending_listing_count'] / Homes['total_listing_count']
Homes.pending_listing_Prop.head()

#Add active_listing_Prop
Homes['active_listing_Prop'] = Homes['active_listing_count'] / Homes['total_listing_count']
Homes.active_listing_Prop.head()


HomesEnriched = Homes[['year', 'month', 'county', 'state', 'median_listing_price', 'active_listing_Prop', 'median_days_on_market', 'new_listing_Prop', 
               'price_increased_Prop', 'price_reduced_Prop', 'pending_listing_Prop', 'median_square_feet', 'average_listing_price',
               'total_listing_count']] 
HomesEnriched.head()
HomesEnriched.tail()


#Find missing Values
mean = HomesEnriched.mean(numeric_only=True)
HomesEnriched.fillna(value=mean, inplace=True)


HomesEnriched['median_days_on_market'] = HomesEnriched['median_days_on_market'].astype(int)
print(type('median_days_on_market'))


print("Unique MDOM Count: " +str(HomesEnriched.median_days_on_market.nunique()))
print(HomesEnriched.median_days_on_market.value_counts())

#Create a new column for the median_market_time
def conditions(s):
    if s['median_days_on_market'] <= 45:
        return 'Rapid'
    if 46 <= s['median_days_on_market'] <= 90:
        return 'Normal'
    if s['median_days_on_market'] in [91, 145]:
        return 'Long'
    else: 
        return 'Stale'

HomesEnriched['median_market_time'] = HomesEnriched.apply(conditions, axis=1)
HomesEnriched.head()


#change the create new column from month using the as four seasons
def conditions(row):
    month = int(row['month'])  # Convert to int here
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# Convert the entire column before applying the function
HomesEnriched['month'] = HomesEnriched['month'].str.replace('o', '').astype(int)  # Convert the entire column before applying the function
HomesEnriched['season'] = HomesEnriched.apply(conditions, axis=1)
HomesEnriched.season.shape
HomesEnriched.tail()


#Prepare the data 
#dummy variables
HomesEnriched_dummies = pd.get_dummies(HomesEnriched, columns=['season','median_market_time'])
HomesEnriched_dummies.head()



#find where Input contains infinity or a value too large for dtype('float64').
HomesEnriched_dummies.replace([np.inf, -np.inf], np.nan, inplace=True)
HomesEnriched_dummies.fillna(0, inplace=True)



# scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
HomesEnriched_dummies[['median_listing_price', 'active_listing_Prop', 'median_days_on_market', 'new_listing_Prop', 'price_increased_Prop', 
               'price_reduced_Prop', 'pending_listing_Prop', 'median_square_feet', 'average_listing_price', 'total_listing_count']] = scaler.fit_transform(HomesEnriched_dummies[['median_listing_price', 'active_listing_Prop', 'median_days_on_market', 'new_listing_Prop', 'price_increased_Prop', 
               'price_reduced_Prop', 'pending_listing_Prop', 'median_square_feet', 'average_listing_price', 'total_listing_count']])

#change string to numeric
HomesEnriched_dummies.columns


#find the best features
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import chi2

# split the data
X = HomesEnriched_dummies.drop(['year', 'month', 'county', 'state', 'median_listing_price'], axis=1)
y = HomesEnriched_dummies['median_listing_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# find the best features
selector_f = SelectKBest(f_regression, k=5)
X_new = selector_f.fit_transform(X_train, y_train)

# get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector_f.inverse_transform(X_new), 
                                 index=X_train.index, 
                                 columns=X_train.columns)
selected_columns = selected_features.columns[selected_features.var() != 0]
selected_columns




# split the data
X = HomesEnriched_dummies[selected_columns]
y = HomesEnriched_dummies['median_listing_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#Use a fitted multiple regression model to make predictions.
# Create a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'MAE: {mae}')
print(f'R2: {r2}')



# Create a polynomial regression model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, y_train)
model = LinearRegression()
model.fit(X_poly, y_train)

# Make predictions
y_pred = model.predict(poly.fit_transform(X_test))

# Evaluate the model
poly_mse = metrics.mean_squared_error(y_test, y_pred)
poly_rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
poly_mae = metrics.mean_absolute_error(y_test, y_pred)
poly_r2 = metrics.r2_score(y_test, y_pred)

print(f'MSE: {poly_mse}')
print(f'RMSE: {poly_rmse}')
print(f'MAE: {poly_mae}')
print(f'R2: {poly_r2}')


# create a table with the results include the result using the pipeline
results = pd.DataFrame({'Model': ['Linear Regression', 'Polynomial Regression', 'Pipeline'],
                        'MSE': [mse, poly_mse, pipeline_mse],
                        'RMSE': [rmse, poly_rmse, pipeline_rmse],
                        'MAE': [mae, poly_mae, pipeline_mae],
                        'R2': [r2, poly_r2, pipeline_r2]})
results



