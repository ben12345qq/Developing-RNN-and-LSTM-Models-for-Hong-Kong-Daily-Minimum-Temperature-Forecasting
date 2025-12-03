

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

# Inspect the file
df = pd.read_csv('.\daily_HKO_GMT_ALL.csv', skiprows=2, encoding='utf-8')
print(df.head(5))
df.columns = [col.strip() for col in df.columns]

# Rename to standard English lowercase
df = df.rename(columns={
    '年/Year': 'year',
    '月/Month': 'month',
    '日/Day': 'day',
    '數值/Value': 'value',
    '數據完整性/data Completeness': 'data_completeness'
})

# Convert to numeric, coerce invalid to NaN (handles '***', footer strings, etc.)
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['month'] = pd.to_numeric(df['month'], errors='coerce')
df['day'] = pd.to_numeric(df['day'], errors='coerce')
df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Handles '***' in temperatures

# Drop rows with NaN in date columns (removes footer and invalid dates)
df = df.dropna(subset=['year', 'month', 'day'])

# Now safely cast date components to integers
df['year'] = df['year'].astype(int)
df['month'] = df['month'].astype(int)
df['day'] = df['day'].astype(int)

# Create datetime index
df['Date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='raise')  # Raise if any issues remain
df = df.set_index('Date')

# Filter for complete data only and select temperature (excludes incomplete or missing values)
df = df[df['data_completeness'] == 'C']['value']

# Handle any remaining missing temperatures (though 'C' should mean no misses)
df = df.interpolate(method='linear').ffill().bfill()

# Subsets
train_data = df['1980-01-01':'2024-12-31']
test_data = df['2025-01-01':'2025-10-30']

# Verify
print(f"Train data shape: {train_data.shape}")  # Should be ~16436 days (45 years, accounting for leap years/missing)
print(f"Test data shape: {test_data.shape}")  # Should be ~304 days (Jan-Oct 2025)
print(train_data.head())
print(test_data.tail())

# Handle missing values
train_data = train_data.interpolate(method='linear').ffill().bfill()  # Interpolate, then fill edges

# Plot for analysis
plt.figure(figsize=(12, 6))
plt.plot(train_data)
plt.title('HK Daily Grass Min Temperature (1980-2024)')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.show()

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_scaled = scaler.transform(test_data.values.reshape(-1, 1))  # Use same scaler

# Create sequences (e.g., window_size=30)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 30  # Experiment with 30-60
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# Reshape for RNN/LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
test_data.to_pickle('test_data.pkl')
with open('window_size.txt', 'w') as f:
    f.write(str(window_size))
# RNN Model
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(window_size, 1), return_sequences=True))
rnn_model.add(SimpleRNN(50))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(window_size, 1), return_sequences=True))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')



# Train (example for LSTM)
early_stop = EarlyStopping(monitor='val_loss', patience=100)
history_lstm = lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])
early_stop = EarlyStopping(monitor='val_loss', patience=100)

history_rnn = rnn_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])




# Save models
rnn_model.save('rnn_model.keras')
lstm_model.save('lstm_model.keras')
