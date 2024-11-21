# %%
#Build - Train - Test Recurrent Neural Networks
#Using Sine Wave Data with Simple RNN & Keras

# %%
# Import basic libraries
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt

#hide warnings
import warnings
warnings.filterwarnings('ignore')

# %%
# from timeseries RNN neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

# %%
# imprt Keras: Time Series Data Preparation
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

#import Keras: min-max scaler
from sklearn.preprocessing import MinMaxScaler

# %%
#Generate Date

# %%
# create a simple sine wave using Numpy
x = np.linspace(0,64,1024)
y= np.sin(x)

# %%
x

# %%
y

# %%
plt.plot(x,y)

# %%
df = pd.DataFrame(data=y, index=x, columns=['Sine'])

# %%
df.head()

# %%
len(df)

# %%
# Split Data --> Train/ Test

# %%
# Set Percentage of data to be used for training
train_percent = 0.2

# %%
# Number of data points reserved for training the model
# 20% of the original dataset

len(df)*train_percent



# %%
# Need to find the index location of the split

test_length = np.round(len(df)*train_percent)

# %%
test_length

# %%
# The testing data set starts at this index

test_start_index = int(len(df) - test_length)

# %%
test_start_index

# %%
# create separate training and testing datasets
data_train = df.iloc[:test_start_index]

# The testing data set starts at this index
data_test = df.iloc[test_start_index:]

# %%
data_train.head()

# %%
data_test.head()

# %%
# create a MinMaxScaler object to normalize the data
scaler = MinMaxScaler()

# %%
#Train the scaler on the training data
scaler.fit(data_train)

# %%
# Normalize the training data
normalized_train = scaler.transform(data_train)

# Normalize the testing data
normalized_test = scaler.transform(data_test)

# %% [markdown]
# #### Create Timeseries Generator Instance

# %%
# set the length of the output sequences
length = 50

# batch size: number of timeseries samples in each batch
batch_size = 1

# create a TimeseriesGenerator object: train_tsGenerator

train_tsGenerator = TimeseriesGenerator(normalized_train, normalized_train, length=length, batch_size=batch_size)

# %%
len(normalized_train)

# %%
# what does the first batch look like?
X,y = train_tsGenerator[0]

# %%
#  Print x.flatten()
X.flatten()

# %%
# print y: waht does X predict?
y

# %% [markdown]
# #### Build Simple RNN Model

# %%
# Data set: Only one column/attribute: Sine values of index x
n_features = 1

# define the model
model = Sequential()

# Add a SimpleRNN layer : Using Simpe RNN cells

model.add(SimpleRNN(100, input_shape=(length, n_features)))

# Add a Dense layer with one neuron

model.add(Dense(1))


# %% [markdown]
# #### Compile Model

# %%
# Compile the model
# Loss function: Mean Squared Error
# Note: Why MES? the data is real values/continuous: A regression problem
# Optimizer: Adam

model.compile(optimizer='adam', loss='mse')

# Train the model
model.summary()

# %% [markdown]
# ### Train (Fit) Model

# %%
# fit the model
# use fit_generator instead of fit

model.fit_generator(train_tsGenerator, epochs=5)

# %%
# Load the Loss data into a DataFrame
df_model_loss = pd.DataFrame(model.history.history)

# Plot the Loss data

df_model_loss.plot()

# %% [markdown]
# 

# %%
length

# %%
# Take a sneak peak at the test data
first_eval_batch = normalized_train[-length:]

first_eval_batch

# %%
# reshape the data to the format required by the model
first_eval_batch = first_eval_batch.reshape((1, length, n_features))

first_eval_batch

# %%
first_eval_batch.shape

# %% [markdown]
# #### Evaluate Model

# %%
# All the Code for evaluation

# store the predictions
test_predictions = []

# last n points from the training set
first_eval_batch = normalized_train[-length:]

# reshape the data to the format required by the model
current_batch = first_eval_batch.reshape((1, length, n_features))

# Run a loop to predict the future values
for i in range(len(data_test)):
    
    # get the prediction
    current_pred = model.predict(current_batch)[0]
    
    # store the prediction
    test_predictions.append(current_pred)
    
    # update the current batch to include the prediction
    # drop the first value and add the prediction
    # append the current prediction to the current batch
    # maintain the length of the batch
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)

# %%
# Inverse Transform the predictions
true_predictions = scaler.inverse_transform(test_predictions)

# Add the predictions to the test data
true_predictions

# %%
data_test

# %%
#create a new column in the test data set

data_test['Predictions'] = true_predictions

# %%
# Plot the data
data_test.plot(figsize=(12,8))


