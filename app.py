import streamlit as st
import google.generativeai as genai
import os
import re
import yfinance as yf  # For validating stock tickers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key is not set")
genai.configure(api_key=api_key)

# Function to get response from Gemini API
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text

# Function to extract ticker and days
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
    days = int(days_match.group(1)) if days_match else 7  # Default to 7 days

    return ticker, days

# Function to validate ticker
def is_valid_ticker(ticker):
    """Checks if the given ticker exists using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return "longName" in info  # If company name exists, ticker is valid
    except:
        return False

# Streamlit UI
st.title("Stock Forecast")

# Search bar for stock input
user_input = st.text_input("Enter your stock forecast request")

if st.button("Get Prediction Details"):
    if user_input:
        ticker, days = extract_ticker_and_days(user_input)

        if ticker == "NOT_FOUND" or not is_valid_ticker(ticker):
            st.error("❌ No company found with this name or invalid stock ticker. Please enter a valid stock symbol.")
        else:
            st.write(f"**✅ Ticker:** {ticker}")
            st.write(f"**📅 Days to Predict:** {days}")
    else:
        st.warning("⚠️ Please enter a query.")
