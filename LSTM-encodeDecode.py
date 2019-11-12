# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:53:52 2019

@author: session1
"""

# univariate multi-step encoder-decoder lstm example
from numpy import array
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

# define input sequence
raw_seq=[]
# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
#conver df to np.array which nn can accept
training_set = dataset_train.iloc[:, 1:2]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set = sc.fit_transform(training_set)

# define input sequence
raw_seq = training_set.tolist()
raw_seq=np.reshape(raw_seq,(1258))
print(raw_seq)



# choose a number of time steps
n_steps_in, n_steps_out = 60, 20
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
y = y.reshape((y.shape[0], y.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(50, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=100,batch_size = 32)

x_input=training_set[-60:]
x_input = array(x_input)
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input)
yhat=yhat.reshape(20,1)
print(yhat)

yhat = sc.inverse_transform(yhat)
print(yhat)

