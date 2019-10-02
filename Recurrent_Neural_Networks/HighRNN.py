import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_dataset = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = train_dataset.iloc[:,2:3].values # high stock prices

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

x_train = []
y_train = []
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
x_train,y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dropout

regressor = Sequential()

#adding the first LSTM layer to avoid the over fitting we use dropout regularization
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(rate= .2))

#adding the second LSTM layer to avoid the over fitting we use dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(rate = .2))

#adding the third LSTM layer to avoid the over fitting we use dropout regularization
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(rate = .2))

#adding the fourth LSTM layer to avoid the over fitting we use dropout regularization
regressor.add(LSTM(units = 50,return_sequences=False))
regressor.add(Dropout(rate = .2))

#adding the output layer
regressor.add(Dense(units = 1))

#compiling the rnn
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting the rnn to the training set
regressor.fit(x_train, y_train,epochs = 100,batch_size = 32)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017(test set)
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 2:3].values

dataset_total = pd.concat((train_dataset['Open'],dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60 :].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_prices = regressor.predict(x_test)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices) #to get the originl val before scaling scaled val

#visualing the results
plt.plot(real_stock_price, color = 'black', label = "real_stock_price")
plt.plot(predicted_stock_prices, color = 'b', label = "predicted_stock_price")
plt.title("Google Stock Prices")
plt.xlabel("Time")
plt.ylabel("Google Stock Prices(2017)")
plt.legend()
plt.show()