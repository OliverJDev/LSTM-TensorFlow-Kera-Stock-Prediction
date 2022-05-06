import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.metrics import Accuracy, Precision
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import tensorflow as tf
#checks availability of gpu for CUDA
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#CUDA_VISIBLE_DEVICES="" - disables gpu (env)
#---------------------------------------------
# DATA PREPARATION
#---------------------------------------------
#Opens the stock's CSV file which contains data about the stock
df = pd.read_csv('AAPL_long.csv')['Open'].values
#converts the dataframe into a single dimension array
df = df.reshape(-1, 1)
#converts 80% of the data into a training array, and 20% into a testing array
dataset_train = np.array(df[:int(df.shape[0]*0.8)])
dataset_test = np.array(df[int(df.shape[0]*0.8):])
#a sklearn function to transform each features by scaling each feature given a range
#the scaling is converting wider range data into a more close nit one
scaler = MinMaxScaler(feature_range=(0,1))
dataset_train = scaler.fit_transform(dataset_train)
dataset_test = scaler.transform(dataset_test)

#creates the training and testing dataset's
def create_dataset(df):
    x = []
    y = []
    #df.shape[0] = the amount of features
    for i in range(50, df.shape[0]):
        #creates a feature and label (x=input(feature)) (y=output(label))
        #appending the last 50 prices for the features
        x.append(df[i-50:i, 0])
        #appending the next price
        y.append(df[i, 0])
    x = np.array(x)
    y = np.array(y)
    return x,y
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)

#---------------------------------------------
# THE MODEL BEING CREATED
#---------------------------------------------
# LSTM requires a 3d array in order to use it in it's layers
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

mse = []
def trainModel(x_train, y_train, epochs, batch):
    #initalize the model (sequential)  - https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    model = Sequential()
    #96 units of ouputs dimensionality, return sequences to true to allow for 3D input
    model.add(LSTM(units=96, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    #drop outs prevent overfitting, dropouts make the effect of noise, and remove random outputs
    model.add(Dropout(0.2))
    #stack's more lstm layers, which makes the model deeper ie more accurate
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=96))
    model.add(Dropout(0.2))
    #a dense layer
    model.add(Dense(units=1))


    #compiles the model.
    #loss is mean_squared_error since we are predicting a price (regression problem)
    #adam optimizer to update hte networks weights based on the training data iteratively
    model.compile(loss='mape', optimizer='adam', metrics=['mape'])


    # ---------------------------------------------
    # THE TRAINING PHASE
    # ---------------------------------------------

    # starts the training model. This utilises n cycles through the full training dataset
    # n batch_size being the number of training examples used in one single cycle
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch)
    mse.append(history.history['mape'])

    return model

model = trainModel(x_train, y_train, 50, 32)

#---------------------------------------------
# THE PREDICTION PHASE
#----------------------------------------------

#runs the prediction utilising the model
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
#the original prices
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

predicted_price = predictions

#---------------------------------------------
# FORECASTING
# This section utilises the last 50 values to generate a new value, then this
# job is repeated with the new value as the new entry, removing the old first value
# generates the next 50 points. This re-trains the data every single iteration
#----------------------------------------------
new_ticks = []
new_ticks.append(pd.read_csv('AAPL_long.csv')['Open'].values[-1])
days_forecast = 0
for x in range(days_forecast):
    df = pd.read_csv('AAPL_long.csv')['Open'].values
    df = df.reshape(-1, 1)
    dataset_test = np.array(df[int(df.shape[0]*0.8):])

    #append newly predicted ticks to the end
    dataset_test = np.append(dataset_test, new_ticks)
    dataset_test = dataset_test.reshape(-1, 1)
    dataset_test = np.array(dataset_test)
    dataset_test = scaler.transform(dataset_test)
    dataset_test = dataset_test[x:]
    x_test, y_test = create_dataset(dataset_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #re-train the model
    model = trainModel(x_test, y_test, 10, 50)
    #make prediction again
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    print("Predicted Next Value - ", predictions[-1:])
    new_ticks.append(predictions[-1:][0][0])
    print("Iterations Left: ", (10 -x), "/10")
    print("New Ticks ", new_ticks)
    print("----")

#---------------------------------------------
# METRICS
#----------------------------------------------


#---------------------------------------------
# VISUALISING THE RESULTS
#----------------------------------------------
#uses matplotlib to visualise the dataset into a graph
fig, ax = plt.subplots(figsize=(16,8))
ax.set_facecolor('#000041')
#The original values
vals = pd.read_csv('AAPL_long.csv')['Open'].values
ax.plot(y_test_scaled, color='red', label='Original price')
plt.xlabel("x")
plt.ylabel("y")
plt.plot(predicted_price, color='cyan', label='Predicted price')
#print(len(mse))

#print(mse[0])


start = len(predictions)
yNewTicks = []
for x in range(start -1, len(new_ticks) + (start -1)):
    yNewTicks.append(x)
plt.plot(yNewTicks , new_ticks,  color='green', label='Predicted Price Forecast')
plt.legend()
plt.show()
