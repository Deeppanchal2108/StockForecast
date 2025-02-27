import streamlit as st
import google.generativeai as genai
import os
import re
import yfinance as yf
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
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

def extract_ticker_and_days(user_input):
    prompt = f"""
    Extract the valid stock ticker symbol and number of future prediction days from this text:
    
    "{user_input}"
    
    If the ticker is not found or invalid, return:
    Ticker: NOT_FOUND  
    Days: <number> (default to 7 if not mentioned)
    
    Ensure the ticker is in uppercase and a valid stock symbol.
    """

    response = get_gemini_response(prompt)
    ticker_match = re.search(r"Ticker:\s*([A-Z]+)", response)
    days_match = re.search(r"Days:\s*(\d+)", response)

    ticker = ticker_match.group(1) if ticker_match else "NOT_FOUND"
    days = int(days_match.group(1)) if days_match else 7

    return ticker, max(7, min(days, 20))

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        return "longName" in stock.info
    except:
        return False

def predict_future_prices(ticker, days):
    model_path = f"models/{ticker}_lstm.h5"
    scaler_path = f"models/{ticker}_scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    df = yf.download(ticker, period="70d")
    if df.shape[0] < 60:
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
        
        input_seq = np.append(input_seq[1:], predicted_price_scaled, axis=0)
    
    return predictions

st.set_page_config(page_title="Stock Forecast")
st.title("📈 Stock Price Prediction")

user_input = st.text_input("Enter your stock forecast request")

if st.button("Get Prediction Details"):
    if user_input:
        ticker, days = extract_ticker_and_days(user_input)

        if ticker == "NOT_FOUND" or not is_valid_ticker(ticker):
            st.error("❌ Invalid stock ticker. Please enter a valid symbol.")
        else:
            st.subheader(f"✅ Ticker: {ticker}")
            st.write(f"📅 **Predicting for {days} days**")

            with st.spinner("🔄 Fetching predictions... Please wait!"):
                predictions = predict_future_prices(ticker, days)

            if predictions:
                df = pd.DataFrame({"Day": list(range(1, days+1)), "Predicted Price": predictions})
                
                # Display Predictions
                st.subheader("📊 Predicted Prices")
                st.dataframe(df.style.format({"Predicted Price": "{:.2f}"}))

                # Fetch historical data
                df_hist = yf.download(ticker, period="70d")
                actual_prices = df_hist["Close"].values[-60:]
                days_actual = list(range(-60, 0))
                days_future = list(range(1, days+1))

                # Plot
                plt.figure(figsize=(10, 5))
                plt.plot(days_actual, actual_prices, label="Actual Prices (Last 60 Days)", color="blue")
                plt.plot(days_future, predictions, label="Predicted Prices", color="red", linestyle="dashed", marker="o")
                plt.axvline(0, color="gray", linestyle="--")
                plt.xlabel("Days")
                plt.ylabel("Stock Price ($)")
                plt.title(f"Stock Price Prediction for {ticker}")
                plt.legend()
                plt.grid(True)

                st.pyplot(plt)

            else:
                st.error("⚠️ Unable to predict. Model or scaler missing.")

    else:
        st.warning("⚠️ Please enter a query.")
