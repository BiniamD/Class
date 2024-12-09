# LSTM Neural Network

# Import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# for timeseries RNN LSTM neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Import Keras TimeSeriesGenerator
# this class produce time series data for RNN LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# ##### Data set : APPL(Apple stock :01/01/2024-06/30/2019)

# Get Apple stock data from Yahoo Finance from 01/01/2014 to 06/30/2019
import yfinance as yf
apple = yf.Ticker("AAPL")
#Tesla = yf.Ticker("TSLA")
apple_stock = apple.history(start="2014-01-01", end="2019-06-30")
#Tesla_stock = Tesla.history(start="2014-01-01", end="2024-11-22")
#apple_stock.head()
#Tesla_stock.head()
#apple_stock = Tesla_stock

#change date as column 
apple_stock.reset_index(inplace=True)
apple_stock['Date'] = apple_stock['Date'].dt.date   
df_ALL = apple_stock.copy()

# #### Brief Exploratory Data Analysis (EDA)

#print shape of the dataframe
df_ALL.shape

#check data types of the dataframe
df_ALL.dtypes

#statistical summary of the dataframe
df_ALL.describe()

# ### Keep only "Close"

#only get close price
df = df_ALL['Close'].to_frame()

df.head()

df.plot(figsize=(12,8))
plt.show()


# set the length of the input sequence
length60 = 60


length60


len(df)


# set percentage of the data for training
test_percent = 0.1


# set the length of the test data
test_length = len(df) * test_percent
test_length


# #### Split Data --> Train/Test


# round the length of the test data
test_length = np.round(len(df)*test_percent)
test_length


# the testing data set starts at this index

#test_start = int(len(df) - test_length - length60)
split_index = int(len(df) - test_length)
split_index


# split the data into training and testing data
data_train = df.iloc[:split_index]


#data_train = df.iloc[:test_start] 
data_test = df.iloc[split_index-length60:]


# check the shape of the training and testing data
data_train.head(5)


data_train.tail()


scaler = MinMaxScaler()


scaler.fit(data_train)


#normalize the data      
normalized_train = scaler.transform(data_train)


#normalize the data test
normalized_test = scaler.transform(data_test)


# check the shape of the normalized data
batch_size32 = 32
# create a time series generator for the training data
train_tsGenerator60 = TimeseriesGenerator(normalized_train, normalized_train, length=length60, batch_size=batch_size32)


len(normalized_train)


len(train_tsGenerator60)


# what does the first batch look like
X,y = train_tsGenerator60[0]


# print x
print(X)


# ![image.png](attachment:image.png)


# print y
print(y)


# ### Build, Train, and Test Model


# Set number of features

n_features = 1


# define the model

model = Sequential()


# ![image.png](attachment:image.png)


# add LSTM layer
# 50 is the number of neurons
# return_sequence is set to True because we are using another LSTM layer
model.add(LSTM(50, return_sequences=True, input_shape=(length60, n_features)))

# add dropout layer
model.add(Dropout(0.2))

# add LSTM layer with relu activation function
model.add(LSTM(50, return_sequences=True, activation='relu'))

# add dropout layer
model.add(Dropout(0.2))

# add LSTM layer with relu activation function
model.add(LSTM(50, activation='relu'))

# add dense layer
model.add(Dense(1))



# compile the model
model.compile(optimizer='adam', loss='mse')

# fit the model
#model.summary()



# train the model
model.fit(train_tsGenerator60, epochs=100)


# #### Visualize Models Performnce 


# loss history keys 
loss_history_keys = model.history.history.keys()

# model.history.history is a dict
# loss history keys

loss_history_keys


# Load the loss data into a dataframe
losses = pd.DataFrame(model.history.history)

# plot the losses and add correct lables
losses.plot()



# ##### Prediction for testing :  Using TimeseriesGenerator


# Create Time Series Generator for the test data

#bath_size must be 1
batch_size1 = 1

# create a time series generator for the training data
test_tsGenerator60 = TimeseriesGenerator(normalized_test, normalized_test, length=length60, batch_size=batch_size1)


# ##### Predict Future Data Points for Testing


# predict the test data
normalized_prediction = model.predict(test_tsGenerator60)


# print the normalized prediction
normalized_prediction


len(normalized_prediction)


# convert the normalized prediction to the original scale

prediction = scaler.inverse_transform(normalized_prediction)


1244 + 60


# Flatten the prediction 2D array with index 1244 to 1382
prediction = prediction.flatten()
prediction_index = np.arange(1244,1382,step = 1)

# create a dataframe for the prediction
df_prediction = pd.DataFrame(data=prediction, index=prediction_index, columns=['Predictions'])


df_prediction


# ### Visualize Prediction


# plot the prediction next to the test data
ax=data_train.plot(figsize=(12,8))
df_prediction.plot(ax=ax,figsize=(12,8))


# plot the prediction
plt.figure(figsize=(12,8))
plt.plot(df.index, df.values, label='Original Data')
plt.plot(df_prediction.index, df_prediction.values, label='Prediction')
plt.legend()



# #### Time Series Forecasting with RNN LSTM Neural Network


## still use minmaxscaler to normalize the data

full_scaler = MinMaxScaler()
normalized_full_data = full_scaler.fit_transform(df)


# #### create a time series generator for the full data


# number of steps of the input time series
# length of the input time series is 60
length60


# Create Time Series Generator for forecasting the future data
# bath_size must be 1

forecast_tsGenerator = TimeseriesGenerator(normalized_full_data, normalized_full_data, length=length60, batch_size=batch_size32)


# ## build, compile and train the model


# train/fit the model
model.fit(forecast_tsGenerator, epochs=100)


# forecast the future data
forecast = []

# get the last 60 days of the normalized data
period = 117
first_eval_batch = normalized_full_data[-length60:]
current_batch = first_eval_batch.reshape((1, length60, n_features))

# forecast the future data
for i in range(period):
    # get the prediction
    current_pred = model.predict(current_batch)[0]
    # append the prediction to the forecast list
    forecast.append(current_pred)
    # update the current batch
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)


# ### Inverse the forecast data to the original scale


# convert the forecast list to an array
forecast = full_scaler.inverse_transform(forecast)

# print the forecast
forecast


df


# create a dataframe for the forecast
forecast_index = np.arange(1382,1499,step = 1)

df_forecast = pd.DataFrame(data=forecast, index=forecast_index, columns=['Forecast'])

df_forecast


# ### Plot the forecast


df.plot()
df_forecast.plot()


# plot the forecast
plt.figure(figsize=(12,8))
plt.plot(df.index, df.values, label='Original Data')
plt.plot(df_prediction.index, df_prediction.values, label='Prediction')
plt.plot(df_forecast.index, df_forecast.values, label='Forecast')
plt.legend()



df_ALL_JUL_DEC_2019 = apple.history(start="2019-07-01", end="2019-12-15")
df_ALL_JUL_DEC_2019.head()



#change date as column 
df_ALL_JUL_DEC_2019.reset_index(inplace=True)


#change datetime column to date
df_ALL_JUL_DEC_2019['Date'] = df_ALL_JUL_DEC_2019['Date'].dt.date   
df_ALL_JUL_DEC_2019.head()



# Only get close price and use iloc
df_JUL_DEC_2019 = df_ALL_JUL_DEC_2019.iloc[:,4]


df_JUL_DEC_2019.plot()


df_forecast['Forecast'].values


# Add the forecast to the original data as a Forecast column in the dataframe
df_JUL_DEC_2019 = df_JUL_DEC_2019.to_frame()




# Ensure the forecast values match the length of the index
forecast_values = df_forecast['Forecast'].values

# Truncate or pad the forecast values to match the length of the index
if len(forecast_values) > len(df_JUL_DEC_2019):
	forecast_values = forecast_values[:len(df_JUL_DEC_2019)]
else:
	forecast_values = np.pad(forecast_values, (0, len(df_JUL_DEC_2019) - len(forecast_values)), 'constant', constant_values=np.nan)

# Add the forecast to the original data as Forecast column as dataframe
df_JUL_DEC_2019['Forecast'] = forecast_values


df_JUL_DEC_2019


df_JUL_DEC_2019.plot()
