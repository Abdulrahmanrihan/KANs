import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('stock_trading_data.csv')

data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .90 ))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
        
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

from keras.models import Sequential
from keras.layers import Dense, LSTM

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=20, verbose=2)

from tkan import TKAN, BSplineActivation
import tensorflow as tf

kan_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)),
    TKAN(100, tkan_activations=[BSplineActivation() for _ in range(5)], return_sequences=True),  # Here tkan_activations can be left empty (in that case only one BSplineActivation), BSplineActivation default order is 3 but you can change it like [BSplineActivation(i) for i in range(5)] for example
    TKAN(100, tkan_activations=[BSplineActivation() for _ in range(5)], return_sequences=False), # But you can put any activation function from standard tensorflow or custom (any callable in fact), it should just be putted in a list 
    tf.keras.layers.Dense(y_train.shape[1], activation='linear'),
], name = 'tkan_model')

kan_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
kan_model.summary()

history = kan_model.fit(x_train, y_train, batch_size=1, epochs=20, verbose = True)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

# Get the models predicted price values 
predictions_lstm = model.predict(x_test)
predictions_lstm = scaler.inverse_transform(predictions_lstm)

# Get the root mean squared error (RMSE)
rmse_lstm = np.sqrt(np.mean(((predictions_lstm - y_test) ** 2)))

# Get the models predicted price values 
predictions_kan = kan_model.predict(x_test)
predictions_kan = scaler.inverse_transform(predictions_kan)

# Get the root mean squared error (RMSE)
rmse_kan = np.sqrt(np.mean(((predictions_kan - y_test) ** 2)))

# Plot the data
train = data[1100:training_data_len]
valid = data[training_data_len:]
valid['Predictions_lstm'] = predictions_lstm
valid['Predictions_kan'] = predictions_kan

# Visualize the data
plt.figure(figsize=(16,6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions_lstm', 'Predictions_kan']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
