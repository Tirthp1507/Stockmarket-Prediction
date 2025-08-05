import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Streamlit config
st.set_page_config(page_title="📈 Stock Market Forecasting App", layout="wide")
st.title("📊 Stock Forecasting App")

# 📁 Sample CSV fallback
SAMPLE_PATH = "sample.csv"

# File uploader
uploaded_file = st.file_uploader("📤 Upload a CSV file with Date & Close/Price columns", type=["csv"])

# Load data (sample or uploaded)
def load_data(file, date_col=None, value_col=None):
    df = pd.read_csv(file)
    df.columns = df.columns.str.lower()

    # Auto-detect columns
    if date_col is None:
        date_col = next((col for col in df.columns if 'date' in col or 'time' in col), None)
    if value_col is None:
        value_col = next((col for col in df.columns if any(x in col for x in ['close', 'price', 'value', 'target'])), None)

    if not date_col or not value_col:
        return None, None, None

    df = df[[date_col, value_col]].copy()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)
    df = df.sort_values("ds")

    return df, date_col, value_col

# Forecasting Models
def run_prophet(df, periods):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

def run_arima(df, periods):
    model = ARIMA(df["y"], order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.forecast(steps=periods)

def run_sarima(df, periods):
    model = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=periods)

def run_lstm(df, periods):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['y'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    input_seq = scaled_data[-60:].reshape(1, 60, 1)
    predictions = []

    for _ in range(periods):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    predicted = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predicted

# Load file (uploaded or sample)
if uploaded_file:
    file = uploaded_file
    st.success("✅ Using uploaded file.")
else:
    file = SAMPLE_PATH
    st.info("ℹ️ No file uploaded. Using sample dataset.")

# Read data
raw_df = pd.read_csv(file)
raw_df.columns = raw_df.columns.str.lower()

# Column selection
st.subheader("📋 Data Preview & Column Selection")
st.write("Choose which columns to use:")

date_col = st.selectbox("Select Date Column", options=raw_df.columns, index=0)
value_col = st.selectbox("Select Price/Value Column", options=raw_df.columns, index=1)

# Re-load with selected columns
if uploaded_file:
    uploaded_file.seek(0)
df, selected_date_col, selected_value_col = load_data(file, date_col, value_col)

if df is None or df.empty:
    st.error("❌ Could not parse selected columns. Please check your file.")
else:
    st.success(f"✅ Using '{selected_date_col}' as Date and '{selected_value_col}' as Value")

    st.subheader("📄 Cleaned Data Preview")
    st.dataframe(df.tail())

    # Forecast model and period
    model_choice = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "SARIMA", "LSTM"])
    periods = st.slider("Forecast Days", min_value=1, max_value=365, value=30)

    if st.button("📈 Run Forecast"):
        st.info(f"Running {model_choice} model...")

        if model_choice == "Prophet":
            forecast, model = run_prophet(df, periods)
            st.subheader("📊 Forecasted Data (Prophet)")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            st.subheader("📉 Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

        else:
            future_dates = pd.date_range(start=df["ds"].max(), periods=periods + 1, freq='D')[1:]

            if model_choice == "ARIMA":
                forecast = run_arima(df, periods)
            elif model_choice == "SARIMA":
                forecast = run_sarima(df, periods)
            elif model_choice == "LSTM":
                forecast = run_lstm(df, periods)

            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "y": forecast
            })

            st.subheader(f"📊 Forecasted Data ({model_choice})")
            st.dataframe(forecast_df.tail())

            st.subheader("📉 Forecast Plot")
            plt.figure(figsize=(10, 5))
            plt.plot(df["ds"], df["y"], label="Historical")
            plt.plot(forecast_df["ds"], forecast_df["y"], label="Forecast", linestyle="--")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.title(f"{model_choice} Forecast")
            plt.legend()
            st.pyplot(plt.gcf())
