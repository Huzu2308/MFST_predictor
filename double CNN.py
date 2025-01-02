import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
import streamlit as st

# Title and description
st.title("Microsoft Stock Price Prediction")
st.write("""
### Predict the next 365 days of stock prices using a Hybrid CNN model.
This application uses past data to predict future prices.
""")

# File path for the dataset
file_path = "MSFT(2000-2023).csv"

try:
    # Load the dataset
    stock_data = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime and filter the required date range
    stock_data['Date'] = pd.to_datetime(stock_data['Date'], errors='coerce')  # Handles invalid dates
    stock_data = stock_data.dropna(subset=['Date'])  # Drop rows with invalid dates
    stock_data_filtered = stock_data[(stock_data['Date'] >= '2000-01-01') & (stock_data['Date'] <= '2023-12-31')]

    # Select and scale the 'Close' prices
    close_prices = stock_data_filtered['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Create the training dataset with a lookback period of 60 days
    lookback = 60  # Number of past days to consider
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])  # Use the past 'lookback' days
        y.append(scaled_data[i, 0])  # Predict the next day
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for CNN input

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train Hybrid (CNN + Dense) model
    model_hybrid = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),  # First CNN layer
        Conv1D(64, kernel_size=3, activation='relu'),  # Second CNN layer
        Flatten(),  # Flatten the output for Dense layers
        Dense(25, activation='relu'),  # Dense hidden layer
        Dense(1)  # Output layer
    ])

    # Compile and train the model
    st.write("### Training the model...")
    model_hybrid.compile(optimizer='adam', loss='mean_squared_error')
    model_hybrid.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=2, batch_size=1)

    # Predict the next 365 days using the last 'lookback' days
    last_known_data = scaled_data[-lookback:]  # Ensure size is exactly 'lookback'
    predicted_prices = []

    for _ in range(365):  # Predict the next 365 days
        input_data = last_known_data.reshape(1, lookback, 1)  # Reshape for CNN input
        prediction = model_hybrid.predict(input_data, verbose=0)  # Suppress verbose output
        predicted_prices.append(prediction[0])  # Add predicted price to the list
        last_known_data = np.append(last_known_data[1:], prediction).reshape(-1, 1)  # Slide window

    # Convert predictions back to the original scale
    predicted_prices_rescaled = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Generate future dates for the next year (365 days)
    last_date = stock_data_filtered['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365)

    # Plot the results
    st.write("### Prediction Results")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(stock_data_filtered['Date'], stock_data_filtered['Close'], label='Actual Prices', color='blue')
    ax.plot(future_dates, predicted_prices_rescaled, label='Predicted Prices', color='green', linestyle='-')
    ax.set_title("Microsoft Stock Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

except Exception as e:
    st.error(f"An error occurred: {e}")
