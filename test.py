import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pickle

rnn_model = load_model('rnn_model.keras')  
lstm_model = load_model('lstm_model.keras') 

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
test_data = pd.read_pickle('test_data.pkl')
with open('window_size.txt', 'r') as f:
    window_size = int(f.read())
# Predict
rnn_preds = rnn_model.predict(X_test)
lstm_preds = lstm_model.predict(X_test)

# Inverse scale
rnn_preds = scaler.inverse_transform(rnn_preds)
lstm_preds = scaler.inverse_transform(lstm_preds)

y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics for both models
rnn_mae = mean_absolute_error(y_test_inv, rnn_preds)
rnn_rmse = np.sqrt(mean_squared_error(y_test_inv, rnn_preds))
lstm_mae = mean_absolute_error(y_test_inv, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_test_inv, lstm_preds))

print(f'RNN: MAE={rnn_mae:.2f}°C, RMSE={rnn_rmse:.2f}°C')
print(f'LSTM: MAE={lstm_mae:.2f}°C, RMSE={lstm_rmse:.2f}°C')

# Plot Actual vs Predicted for RNN
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[window_size:], y_test_inv, label='Actual')
plt.plot(test_data.index[window_size:], rnn_preds, label='RNN Predicted')
plt.legend()
plt.title('RNN: Actual vs Predicted (2025 Test Data)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.savefig('rnn_actual_vs_predicted.png')  # Save for report/GitHub
plt.show()

# Plot Actual vs Predicted for LSTM
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[window_size:], y_test_inv, label='Actual')
plt.plot(test_data.index[window_size:], lstm_preds, label='LSTM Predicted')
plt.legend()
plt.title('LSTM: Actual vs Predicted (2025 Test Data)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.savefig('lstm_actual_vs_predicted.png')  # Save for report/GitHub
plt.show()

# Error distribution for LSTM (example; repeat for RNN if needed)
errors = y_test_inv - lstm_preds
plt.hist(errors, bins=30)
plt.title('LSTM Error Distribution')
plt.xlabel('Error (°C)')
plt.ylabel('Frequency')
plt.savefig('lstm_error_distribution.png')
plt.show()
# Error distribution for RNN (example; repeat for RNN if needed)
errors = y_test_inv - rnn_preds
plt.hist(errors, bins=30)
plt.title('RNN Error Distribution')
plt.xlabel('Error (°C)')
plt.ylabel('Frequency')
plt.savefig('rnn_error_distribution.png')
plt.show()






# Optional: Confidence intervals (simple example using rolling std dev on predictions)
rolling_std = pd.Series(lstm_preds.flatten()).rolling(window=7).std().values  # 7-day rolling std
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[window_size:], y_test_inv, label='Actual')
plt.plot(test_data.index[window_size:], lstm_preds, label='LSTM Predicted')
plt.fill_between(test_data.index[window_size:], lstm_preds.flatten() - rolling_std, lstm_preds.flatten() + rolling_std, color='gray', alpha=0.2, label='Confidence Interval (±1 std)')
plt.legend()
plt.title('LSTM with Confidence Intervals (2025 Test Data)')
plt.savefig('lstm_with_ci.png')
plt.show()