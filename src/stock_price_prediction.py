# Step 1: Install necessary dependencies
# Run the following command in your terminal to install dependencies:
# pip install pandas numpy matplotlib scikit-learn tensorflow openpyxl

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential  # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import LSTM, GRU, Dense, SimpleRNN  # pyright: ignore[reportMissingImports]

# Initialize a dictionary to store the results
results = {
    'Algorithm': ['Linear Regression', 'RNN', 'LSTM', 'GRU'],
    'MAE': [],
    'RMSE': []
}

# Step 2: Load the dataset
df = pd.read_csv('Datafile.csv')  # Replace 'Datafile.csv' with your actual file path

# Convert Date to datetime and sort by date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Handle missing values (drop any rows with missing values)
df = df.dropna()

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
df[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close']])

# Step 3: Feature Engineering (Sliding Window Approach)
# Create features and targets
X = df[['Open', 'High', 'Low']].values  # Features (Open, High, Low)
y = df['Close'].values  # Target (Close)

def create_sequences(data, target, time_steps):
    X_data, y_data = [], []
    for i in range(len(data) - time_steps):
        X_data.append(data[i:i + time_steps])
        y_data.append(target[i + time_steps])
    return np.array(X_data), np.array(y_data)

# Set time_steps (number of previous days used to predict the next day's stock price)
time_steps = 60  # You can adjust this value

# Create sequences
X, y = create_sequences(X, y, time_steps)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Regression Model (Linear Regression)
# Fit a simple Linear Regression model
regressor = LinearRegression()

# Reshape X_train for Linear Regression (flatten the time_steps and features)
X_train_2d = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

# Fit the model
regressor.fit(X_train_2d, y_train)

# Predict on test data
X_test_2d = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])  # Reshape for test data
y_pred_reg = regressor.predict(X_test_2d)

# Denormalize the predictions and actual values for MAE and RMSE calculation
y_test_denorm = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 3)), y_test.reshape(-1, 1)), axis=1))[:, -1]
y_pred_reg_denorm = scaler.inverse_transform(np.concatenate((np.zeros((y_pred_reg.shape[0], 3)), y_pred_reg.reshape(-1, 1)), axis=1))[:, -1]

# Calculate MAE and RMSE on the original scale
mae_reg = mean_absolute_error(y_test_denorm, y_pred_reg_denorm)
rmse_reg = np.sqrt(mean_squared_error(y_test_denorm, y_pred_reg_denorm))
print(f'Regression MAE: {mae_reg}, RMSE: {rmse_reg}')

# Append results for Linear Regression
results['MAE'].append(mae_reg)
results['RMSE'].append(rmse_reg)

# Plot actual vs predicted prices (for Regression)
plt.figure(figsize=(12,6))
plt.plot(y_test_denorm, label='Actual Prices')
plt.plot(y_pred_reg_denorm, label='Predicted Prices (Linear Regression)')
plt.legend()
plt.title('Stock Price Prediction - Regression')
plt.show()

# Step 5: Build and Train RNN Model
model_rnn = Sequential()
model_rnn.add(SimpleRNN(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model_rnn.add(Dense(units=1))  # Output layer for predicting stock prices

model_rnn.compile(optimizer='adam', loss='mean_squared_error')
model_rnn.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict on test data
y_pred_rnn = model_rnn.predict(X_test)

# Denormalize the predictions for RNN
y_pred_rnn_denorm = scaler.inverse_transform(np.concatenate((np.zeros((y_pred_rnn.shape[0], 3)), y_pred_rnn.reshape(-1, 1)), axis=1))[:, -1]

# Calculate MAE and RMSE for RNN
mae_rnn = mean_absolute_error(y_test_denorm, y_pred_rnn_denorm)
rmse_rnn = np.sqrt(mean_squared_error(y_test_denorm, y_pred_rnn_denorm))
print(f'RNN MAE: {mae_rnn}, RMSE: {rmse_rnn}')

# Append results for RNN
results['MAE'].append(mae_rnn)
results['RMSE'].append(rmse_rnn)

# Plot actual vs predicted prices (for RNN)
plt.figure(figsize=(12,6))
plt.plot(y_test_denorm, label='Actual Prices')
plt.plot(y_pred_rnn_denorm, label='Predicted Prices (RNN)')
plt.legend()
plt.title('Stock Price Prediction - RNN')
plt.show()

# Step 6: Build and Train LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm.add(Dense(units=1))  # Output layer for predicting stock prices

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict on test data
y_pred_lstm = model_lstm.predict(X_test)

# Denormalize the predictions for LSTM
y_pred_lstm_denorm = scaler.inverse_transform(np.concatenate((np.zeros((y_pred_lstm.shape[0], 3)), y_pred_lstm.reshape(-1, 1)), axis=1))[:, -1]

# Calculate MAE and RMSE for LSTM
mae_lstm = mean_absolute_error(y_test_denorm, y_pred_lstm_denorm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_denorm, y_pred_lstm_denorm))
print(f'LSTM MAE: {mae_lstm}, RMSE: {rmse_lstm}')

# Append results for LSTM
results['MAE'].append(mae_lstm)
results['RMSE'].append(rmse_lstm)

# Plot actual vs predicted prices (for LSTM)
plt.figure(figsize=(12,6))
plt.plot(y_test_denorm, label='Actual Prices')
plt.plot(y_pred_lstm_denorm, label='Predicted Prices (LSTM)')
plt.legend()
plt.title('Stock Price Prediction - LSTM')
plt.show()

# Step 7: Build and Train GRU Model
model_gru = Sequential()
model_gru.add(GRU(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru.add(Dense(units=1))  # Output layer for predicting stock prices

model_gru.compile(optimizer='adam', loss='mean_squared_error')
model_gru.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict on test data
y_pred_gru = model_gru.predict(X_test)

# Denormalize the predictions for GRU
y_pred_gru_denorm = scaler.inverse_transform(np.concatenate((np.zeros((y_pred_gru.shape[0], 3)), y_pred_gru.reshape(-1, 1)), axis=1))[:, -1]

# Calculate MAE and RMSE for GRU
mae_gru = mean_absolute_error(y_test_denorm, y_pred_gru_denorm)
rmse_gru = np.sqrt(mean_squared_error(y_test_denorm, y_pred_gru_denorm))
print(f'GRU MAE: {mae_gru}, RMSE: {rmse_gru}')

# Append results for GRU
results['MAE'].append(mae_gru)
results['RMSE'].append(rmse_gru)

# Plot actual vs predicted prices (for GRU)
plt.figure(figsize=(12,6))
plt.plot(y_test_denorm, label='Actual Prices')
plt.plot(y_pred_gru_denorm, label='Predicted Prices (GRU)')
plt.legend()
plt.title('Stock Price Prediction - GRU')
plt.show()

# Convert results dictionary to DataFrame
results_df = pd.DataFrame(results)

# Step 8: Save results to Excel
results_df.to_excel('model_results.xlsx', index=False, engine='openpyxl')

# Display the results in the console (optional)
print(results_df)

# Bar chart of the results
plt.figure(figsize=(12, 6))
plt.bar(results_df['Algorithm'], results_df['MAE'], color='blue', alpha=0.6, label='MAE')
plt.bar(results_df['Algorithm'], results_df['RMSE'], color='red', alpha=0.6, label='RMSE')
plt.title('Model Performance Comparison')
plt.xlabel('Model')
plt.ylabel('Error')
plt.legend()
plt.show()

# Step 9: Conclusion
# After running the above models, compare their performance based on MAE, RMSE.
# You can select the best performing model and fine-tune it for better accuracy.
