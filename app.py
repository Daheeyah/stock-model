import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
@st.cache_resource
def load_assets():
    model = load_model("gru_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_assets()

st.title("📈 NIFTY Stock Price Prediction (GRU)")
st.markdown("""
This app uses a **Gated Recurrent Unit (GRU)** deep learning model trained on 
NIFTY 1-minute historical data (2015–2025) to predict the next closing price.

**How it works:** Enter the last **60 closing prices** (one per line or comma-separated). 
The model looks at the past 60 minutes to forecast the price for the next minute.
""")

# ============================================================
# INPUT: 60 Past Prices
# ============================================================
st.subheader("Enter Last 60 Closing Prices")

input_method = st.radio("Choose input method:", ("Paste numbers (comma-separated)", "Upload CSV"))

past_prices = None

if input_method == "Paste numbers (comma-separated)":
    user_input = st.text_area(
        "Paste 60 closing prices here:", 
        placeholder="8250.50, 8251.20, 8249.80, ... (60 values)",
        height=150
    )
    if user_input:
        try:
            past_prices = [float(x.strip()) for x in user_input.split(",") if x.strip()]
        except ValueError:
            st.error("Please enter valid numbers only, separated by commas.")

else:
    uploaded_file = st.file_uploader("Upload CSV with a 'close' column (60 rows)", type=["csv"])
    if uploaded_file is not None:
        import pandas as pd
        df_upload = pd.read_csv(uploaded_file)
        if 'close' in df_upload.columns:
            past_prices = df_upload['close'].tail(60).tolist()
        else:
            st.error("CSV must contain a 'close' column.")

# ============================================================
# PREDICT
# ============================================================
if past_prices is not None and len(past_prices) > 0:
    st.write(f"Received **{len(past_prices)}** prices.")
    
    if len(past_prices) != 60:
        st.warning(f"⚠️ Your model was trained on exactly 60 time steps. You provided {len(past_prices)}. Please provide exactly 60 for accurate results.")
    
    if st.button("Predict Next Price"):
        try:
            # Scale the input
            prices_array = np.array(past_prices).reshape(-1, 1)
            prices_scaled = scaler.transform(prices_array)
            
            # Take last 60 if user gave more
            sequence = prices_scaled[-60:].reshape(1, 60, 1).astype('float32')
            
            # Predict
            pred_scaled = model.predict(sequence, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)
            
            st.success(f"### Predicted Next Closing Price: **₹{pred_price[0][0]:,.2f}**")
            
            # Optional: show input chart
            st.line_chart(past_prices)
            
        except Exception as e:
            st.error(f"Prediction failed: {e}")
