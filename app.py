import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.callbacks import EarlyStopping
import datetime
from datetime import date
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide", page_icon="📈")

# Title and description
st.title("📈 Stock Price Prediction: LSTM vs GRU")
st.markdown("""
This app predicts stock prices using **LSTM** and **GRU** neural networks and compares their performance.
""")

# Sidebar for user inputs
st.sidebar.header("⚙️ Configuration")

# Stock selection
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", date(2020, 1, 1))
with col2:
    end_date = st.date_input("End Date", date.today())

# Model parameters
st.sidebar.subheader("Model Parameters")
lookback = st.sidebar.slider("Lookback Period (days)", 10, 100, 60)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)
train_split = st.sidebar.slider("Training Data Split (%)", 60, 90, 80) / 100

# Button to run prediction
predict_button = st.sidebar.button("🚀 Run Prediction", type="primary")

# Function to download stock data
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end, progress=False)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {e}")
        return None

# Function to prepare data
def prepare_data(data, lookback):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Function to create LSTM model
def create_lstm_model(lookback):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create GRU model
def create_gru_model(lookback):
    model = Sequential([
        GRU(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        GRU(units=50, return_sequences=True),
        Dropout(0.2),
        GRU(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2,
        'MAPE': mape
    }

# Main execution
if predict_button:
    if stock_symbol:
        with st.spinner(f"Loading data for {stock_symbol}..."):
            df = load_data(stock_symbol, start_date, end_date)
        
        if df is not None and len(df) > 0:
            # Display stock data
            st.subheader(f"📊 {stock_symbol} Stock Data")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Highest", f"${df['Close'].max():.2f}")
            with col3:
                st.metric("Lowest", f"${df['Close'].min():.2f}")
            with col4:
                change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
                st.metric("Total Change", f"{change:.2f}%")
            
            # Plot historical data
            st.subheader("Historical Stock Price")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df['Close'], label='Close Price', color='blue')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Prepare data
            with st.spinner("Preparing data..."):
                close_prices = df['Close'].values
                X, y, scaler = prepare_data(close_prices, lookback)
                
                # Split data
                split_idx = int(len(X) * train_split)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                st.success(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples")
            
            # Train models
            col1, col2 = st.columns(2)
            
            # LSTM Model
            with col1:
                st.subheader("🧠 LSTM Model")
                with st.spinner("Training LSTM model..."):
                    lstm_model = create_lstm_model(lookback)
                    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                    
                    progress_bar = st.progress(0)
                    history_lstm = lstm_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[early_stop]
                    )
                    progress_bar.progress(100)
                    
                    # Predictions
                    lstm_train_pred = lstm_model.predict(X_train, verbose=0)
                    lstm_test_pred = lstm_model.predict(X_test, verbose=0)
                    
                    # Inverse transform
                    lstm_train_pred = scaler.inverse_transform(lstm_train_pred)
                    lstm_test_pred = scaler.inverse_transform(lstm_test_pred)
                    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
                    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
                    
                    # Calculate metrics
                    lstm_metrics = calculate_metrics(y_test_actual, lstm_test_pred)
                    
                    st.success("✅ LSTM Model Trained!")
                    
                    # Display metrics
                    for metric, value in lstm_metrics.items():
                        st.metric(metric, f"{value:.4f}")
            
            # GRU Model
            with col2:
                st.subheader("🧠 GRU Model")
                with st.spinner("Training GRU model..."):
                    gru_model = create_gru_model(lookback)
                    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                    
                    progress_bar = st.progress(0)
                    history_gru = gru_model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0,
                        callbacks=[early_stop]
                    )
                    progress_bar.progress(100)
                    
                    # Predictions
                    gru_train_pred = gru_model.predict(X_train, verbose=0)
                    gru_test_pred = gru_model.predict(X_test, verbose=0)
                    
                    # Inverse transform
                    gru_train_pred = scaler.inverse_transform(gru_train_pred)
                    gru_test_pred = scaler.inverse_transform(gru_test_pred)
                    
                    # Calculate metrics
                    gru_metrics = calculate_metrics(y_test_actual, gru_test_pred)
                    
                    st.success("✅ GRU Model Trained!")
                    
                    # Display metrics
                    for metric, value in gru_metrics.items():
                        st.metric(metric, f"{value:.4f}")
            
            # Comparison
            st.subheader("📊 Model Comparison")
            
            # Metrics comparison table
            comparison_df = pd.DataFrame({
                'LSTM': lstm_metrics,
                'GRU': gru_metrics
            })
            st.dataframe(comparison_df.style.highlight_min(axis=1, props='color:white; background-color:green;'))
            
            # Determine better model
            lstm_score = sum([lstm_metrics['RMSE'], lstm_metrics['MAE'], lstm_metrics['MAPE']])
            gru_score = sum([gru_metrics['RMSE'], gru_metrics['MAE'], gru_metrics['MAPE']])
            
            if lstm_score < gru_score:
                st.success("🏆 **LSTM** performs better overall!")
            else:
                st.success("🏆 **GRU** performs better overall!")
            
            # Plot predictions
            st.subheader("📈 Prediction vs Actual Prices")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # LSTM Plot
            train_dates = df.index[lookback:split_idx+lookback]
            test_dates = df.index[split_idx+lookback:split_idx+lookback+len(y_test)]
            
            ax1.plot(train_dates, y_train_actual, label='Actual Train', color='blue', alpha=0.6)
            ax1.plot(train_dates, lstm_train_pred, label='LSTM Train Prediction', color='orange', alpha=0.6)
            ax1.plot(test_dates, y_test_actual, label='Actual Test', color='green', linewidth=2)
            ax1.plot(test_dates, lstm_test_pred, label='LSTM Test Prediction', color='red', linewidth=2)
            ax1.set_title(f'LSTM Model - {stock_symbol} Stock Price Prediction')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # GRU Plot
            ax2.plot(train_dates, y_train_actual, label='Actual Train', color='blue', alpha=0.6)
            ax2.plot(train_dates, gru_train_pred, label='GRU Train Prediction', color='orange', alpha=0.6)
            ax2.plot(test_dates, y_test_actual, label='Actual Test', color='green', linewidth=2)
            ax2.plot(test_dates, gru_test_pred, label='GRU Test Prediction', color='red', linewidth=2)
            ax2.set_title(f'GRU Model - {stock_symbol} Stock Price Prediction')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Training history
            st.subheader("📉 Training Loss History")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
            
            ax1.plot(history_lstm.history['loss'], label='LSTM Loss', color='blue')
            ax1.set_title('LSTM Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(history_gru.history['loss'], label='GRU Loss', color='green')
            ax2.set_title('GRU Training Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Future Prediction (Optional)
            st.subheader("🔮 Future Price Prediction")
            future_days = st.slider("Predict next N days", 1, 30, 7)
            
            if st.button("Predict Future Prices"):
                # Use the better performing model
                better_model = lstm_model if lstm_score < gru_score else gru_model
                model_name = "LSTM" if lstm_score < gru_score else "GRU"
                
                last_sequence = X_test[-1]
                future_predictions = []
                
                for _ in range(future_days):
                    next_pred = better_model.predict(last_sequence.reshape(1, lookback, 1), verbose=0)
                    future_predictions.append(next_pred[0, 0])
                    last_sequence = np.append(last_sequence[1:], next_pred)
                
                future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
                
                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                
                # Display predictions
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions.flatten()
                })
                
                st.write(f"**{model_name} Model Predictions:**")
                st.dataframe(future_df)
                
                # Plot
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df.index[-60:], df['Close'].values[-60:], label='Historical', color='blue')
                ax.plot(future_dates, future_predictions, label='Future Prediction', color='red', marker='o')
                ax.set_title(f'{stock_symbol} - Future Price Prediction using {model_name}')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.error("❌ No data found for the specified stock symbol and date range.")
    else:
        st.warning("⚠️ Please enter a stock symbol.")
else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Prediction' to start.")
    
    # Display sample instructions
    st.markdown("""
    ### How to use:
    1. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT, TSLA)
    2. Select date range for historical data
    3. Adjust model parameters (lookback period, epochs, batch size)
    4. Click "Run Prediction" to train and compare models
    
    ### Metrics Explained:
    - **MSE**: Mean Squared Error (lower is better)
    - **RMSE**: Root Mean Squared Error (lower is better)
    - **MAE**: Mean Absolute Error (lower is better)
    - **R2 Score**: Coefficient of determination (higher is better, max 1.0)
    - **MAPE**: Mean Absolute Percentage Error (lower is better)
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Data from Yahoo Finance")
