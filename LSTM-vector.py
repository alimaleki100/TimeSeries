# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:40:14 2019

@author: session1
"""

# univariate multi-step vector-output stacked lstm example
from numpy import array
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
 
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
 
    
raw_seq=[]
# Importing the training set
dataset_train = pd.read_excel('mcitopup.xlsx')
#conver df to np.array which nn can accept
training_set = dataset_train.iloc[:, 1:2]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set = sc.fit_transform(training_set)

# define input sequence
raw_seq = training_set.tolist()
raw_seq=np.reshape(raw_seq,(573))
print(raw_seq)
# choose a number of time steps
n_steps_in, n_steps_out = 60, 20
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

"""
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=50, verbose=0)


"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50,batch_size = 32)


x_input=training_set[-60:]
x_input = np.array(x_input)
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input)
print(yhat)

yhat = sc.inverse_transform(yhat).T
print(yhat)







