import streamlit as st
import google.generativeai as genai
import os
import re
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key is not set")
genai.configure(api_key=api_key)

def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text

# Extract ticker and days from user input
def extract_ticker_and_days(user_input):
    """Extracts stock ticker and number of days from the response."""
    prompt = f"""
    Extract the valid stock ticker symbol and number of future prediction days from this text:
    
    "{user_input}"
    
    If the ticker is not found or invalid, return:
    Ticker: NOT_FOUND  
    Days: <number> (default to 7 if not mentioned)
    
    Ensure the ticker is in uppercase and a valid stock symbol.
    """

    response = get_gemini_response(prompt)

    # Regex to extract ticker and days
    ticker_match = re.search(r"Ticker:\s*([A-Z]+)", response)
    days_match = re.search(r"Days:\s*(\d+)", response)

    ticker = ticker_match.group(1) if ticker_match else "NOT_FOUND"
    days = int(days_match.group(1)) if days_match else 7

    # Ensure days is within the valid range [7, 20]
    days = max(7, min(days, 20))

    return ticker, days

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return "longName" in info
    except:
        return False

def predict_future_prices(ticker, days):
    model_path = f"models/{ticker}_lstm.h5"
    scaler_path = f"models/{ticker}_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Model or scaler missing for {ticker}. Exiting...")
        return None
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # Fetch latest stock data
    df = yf.download(ticker, period="70d")
    if df.shape[0] < 60:
        print(f"Not enough data to predict for {ticker}. Exiting...")
        return None
    
    last_60_days = df["Close"].values[-60:].reshape(-1, 1)
    last_60_days_scaled = scaler.transform(last_60_days)
    
    predictions = []
    input_seq = last_60_days_scaled.copy()
    
    for _ in range(days):
        X_test = np.array([input_seq])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        predicted_price_scaled = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]
        predictions.append(predicted_price)
        
        # Update input sequence for next prediction
        input_seq = np.append(input_seq[1:], predicted_price_scaled, axis=0)
    
    return predictions

# Streamlit UI
st.set_page_config(page_title="Stock Forecast")
st.title("📈 Stock Price Prediction")

user_input = st.text_input("Enter your stock forecast request")

if st.button("Get Prediction Details"):
    if user_input:
        ticker, days = extract_ticker_and_days(user_input)

        if ticker == "NOT_FOUND" or not is_valid_ticker(ticker):
            st.error("❌ No company found with this name or invalid stock ticker. Please enter a valid stock symbol.")
        else:
            st.write(f"**✅ Ticker:** {ticker}")
            st.write(f"**📅 Days to Predict (limited to 7-20):** {days}")
            
            with st.spinner("🔄 Fetching predictions... Please wait!"):
                predictions = predict_future_prices(ticker, days)

            if predictions:
                df = pd.DataFrame({"Day": [f"Day {i+1}" for i in range(len(predictions))], "Predicted Price ($)": predictions})
                st.subheader("📊 Predicted Prices")
                st.dataframe(df.style.format({"Predicted Price ($)": "{:.2f}"}))  # Format prices to 2 decimal places
            else:
                st.error("⚠️ Unable to predict stock prices. Ensure the model and scaler exist for this ticker.")

    else:
        st.warning("⚠️ Please enter a query.")
