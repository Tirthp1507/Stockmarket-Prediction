import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.preprocessing.sequence import pad_sequences
import warnings
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping



st.set_page_config(page_title=" Stock Forecasting", layout="wide")
st.title(" Multi-Model Stock Price Forecasting Dashboard")

# Sidebar - Ticker input
ticker = st.sidebar.selectbox("Select Stock Ticker", ["AAPL", "GOOGL", "MSFT", "META", "AMZN"])
start = st.sidebar.date_input("Start Date", pd.to_datetime("2018-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Load data
df = yf.download(ticker, start=start, end=end)

# Fix for MultiIndex if present
if isinstance(df.columns, pd.MultiIndex):
    data = df.loc[:, (slice(None), ticker)]
    data.columns = data.columns.droplevel(1)

# Drop rows with missing values
data.dropna(inplace=True)

# Stop if no data
if data.empty:
    st.error("No data found for the selected date range.")
    st.stop()

# Dataset Summary Section
st.subheader(" Dataset Summary")

# Show shape
st.write("**Shape of dataset:**", data.shape)

# Show all columns in a single row
st.write("**Columns available in dataset:**")
columns_row = ", ".join([f"`{col}`" for col in data.columns])
st.markdown(columns_row)

# Show top 5 rows of the dataset
st.write("**Preview of dataset (first 5 rows):**")
st.dataframe(data.head())


# Tabs
tabs = st.tabs(["EDA","ARIMA", "SARIMA", "Prophet", "LSTM"])

with tabs[0]:  # EDA tab
    st.subheader(" Exploratory Data Analysis")

    # 1. Close Price Over Time
    st.markdown("### 1. Close Price Over Time")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(data.index, data["Close"], label="Close Price", color='blue')
    ax1.set_title("Close Price Trend")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True)
    st.pyplot(fig1)

    # 2. Moving Averages (MA100 & MA200)
    st.markdown("### 2. Moving Averages (MA100 & MA200)")
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(data.index, data["Close"], label="Close", alpha=0.6)
    ax2.plot(data.index, data["MA100"], label="MA100", color='orange')
    ax2.plot(data.index, data["MA200"], label="MA200", color='green')
    ax2.set_title("Close Price with MA100 & MA200")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price ($)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # 3. Volume Over Time
    st.markdown("###  3. Volume Traded Over Time")
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.bar(data.index, data['Volume'], color='purple')
    ax3.set_title("Trading Volume")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Volume")
    ax3.grid(True)
    st.pyplot(fig3)
    
    # 4. Volatility
    st.markdown("### 4. Volatility (Rolling Std Dev - 30 Days)")
    data["Volatility"] = data["Close"].rolling(window=30).std()
    fig4, ax4 = plt.subplots(figsize=(12, 3))
    ax4.plot(data.index, data["Volatility"], color="purple")
    ax4.set_title(f"{ticker} Rolling 30-Day Volatility")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Volatility")
    ax4.grid(True)
    st.pyplot(fig4)
    
     # 5.Volume Traded Over Time
    st.markdown("### 5.Volume Traded Over Time")
    fig5, ax5 = plt.subplots(figsize=(12, 3))
    ax5.plot(data.index, data["Volume"], color='brown')
    ax5.set_xlabel("Date")
    ax5.set_ylabel("Volume")
    ax5.set_title(f"{ticker} Trading Volume")
    ax5.grid(True)
    st.pyplot(fig5)
    

# ----- ARIMA ----- #
with tabs[1]:
    st.subheader(" ARIMA Forecast (Next 30 Business Days)")

    try:
        # Time series data
        ts = data["Close"].dropna()

        # Fit ARIMA model
        model = auto_arima(ts, seasonal=False, suppress_warnings=True)
        model_fit = model.fit(ts)

        # Forecast next 30 business days
        n_periods = 30
        forecast = model.predict(n_periods=n_periods)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='B')

        # Plot actual + forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ts[-90:], label="Actual", color='blue')  # Last 90 actual
        ax.plot(forecast_index, forecast, label="Forecast", color='red')  # Forecast as red line
        ax.axvline(forecast_index[0], linestyle='--', color='gray', label='Forecast Start')
        ax.set_title("ARIMA Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close Price")
        ax.legend()
        st.pyplot(fig)

        # Evaluation metrics using last 30 actual values
        actual = ts[-30:]
        predicted = model.predict_in_sample(start=len(ts)-30, end=len(ts)-1)

        if len(actual) == len(predicted):
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("MAPE", f"{mape:.2f}%")
            col3.metric("RMSE", f"{rmse:.2f}")
            col4.metric("R²", f"{r2:.2f}")
        else:
            st.warning("Mismatch between actual and predicted.")

    except Exception as e:
        st.error(f"ARIMA Error: {e}")
# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

with tabs[2]:
    st.subheader(" SARIMA Forecast (Next 30 Business Days)")

    try:
        ts = data["Close"].dropna()

        # Fit SARIMA model
        model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
        results = model.fit(disp=False)

        # Forecast next 30 business days
        forecast = results.get_forecast(steps=30)
        forecast_mean = forecast.predicted_mean
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

        # Plot actual + forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ts[-90:], label='Actual', color='blue')
        ax.plot(forecast_index, forecast_mean.values, label='Forecast', color='red')
        ax.axvline(forecast_index[0], linestyle='--', color='gray', label="Forecast Start")
        ax.set_title("SARIMA Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Evaluation Metrics (in-sample prediction)
        actual = ts[-30:]
        predicted = results.predict(start=len(ts)-30, end=len(ts)-1)

        if len(actual) == len(predicted):
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("MAPE", f"{mape:.2f}%")
            col3.metric("RMSE", f"{rmse:.2f}")
            col4.metric("R²", f"{r2:.2f}")
        else:
            st.warning("Mismatch in prediction and actual values!")

    except Exception as e:
        st.error(f"SARIMA Error: {e}")

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

#prophet
with tabs[3]:  # or whatever tab you're using
    st.subheader(" Final Optimized Prophet Forecast")

    try:
        ts = data["Close"].dropna()

        # Use last 300 points for stability
        ts = ts[-300:]

        df = ts.reset_index()
        df.columns = ["ds", "y"]
        df["y_log"] = np.log1p(df["y"])

        # Build Prophet model with tuned hyperparameters
        model = Prophet(
            changepoint_prior_scale=0.05,  # Smooth trend
            seasonality_mode="additive",
            seasonality_prior_scale=10,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        model.fit(df[["ds", "y_log"]].rename(columns={"y_log": "y"}))

        future = model.make_future_dataframe(periods=30, freq="B")
        forecast = model.predict(future)
        forecast["yhat_inv"] = np.expm1(forecast["yhat"])
        forecast.set_index("ds", inplace=True)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ts.plot(ax=ax, label="Actual", color='blue')
        forecast["yhat_inv"].tail(30).plot(ax=ax, label="Forecast", color='red')
        ax.axvline(x=forecast.index[-30], linestyle="--", color="gray", label="Forecast Start")
        ax.set_title("Prophet Forecast (Optimized)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Evaluation — last 60 days overlap only
        overlap_days = min(len(df), 60)
        actual_log = df.set_index("ds")["y_log"][-overlap_days:]
        predicted_log = forecast["yhat"].loc[actual_log.index]

        if len(predicted_log) == len(actual_log):
            actual = np.expm1(actual_log)
            predicted = np.expm1(predicted_log)

            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{mae:.2f}")
            col2.metric("MAPE", f"{mape:.2f}%")
            col3.metric("RMSE", f"{rmse:.2f}")
            col4.metric("R²", f"{r2:.2f}")
        else:
            st.warning("⚠️ Not enough overlapping data for evaluation.")

    except Exception as e:
        st.error(f"Prophet Forecast Error: {e}")

# LSTM Tab
with tabs[4]:
    st.subheader(" LSTM Forecast")

    try:
        # Use last 3 years for training
        ts = data["Close"].dropna()
        df = ts.to_frame().reset_index()
        df.columns = ['ds', 'y']
        df = df.tail(750)

        # Scaling
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['y']])

        # Sequence creation
        time_step = 60  # increased context
        def create_sequences(data, time_step):
            X, y = [], []
            for i in range(time_step, len(data)):
                X.append(data[i - time_step:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Train-test split
        split = int(len(X) * 0.9)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Improved LSTM Model
        from keras.layers import LSTM, Dense, Dropout, Bidirectional

        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_step, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Training
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=10, batch_size=16, callbacks=[es], verbose=0)

        #  Prediction
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Forecast next 30 days
        last_seq = scaled_data[-time_step:]
        input_seq = last_seq.reshape(1, time_step, 1)
        future_preds = []
        for _ in range(30):
            next_pred = model.predict(input_seq)[0]
            future_preds.append(next_pred)
            input_seq = np.append(input_seq[:, 1:, :], [[next_pred]], axis=1)

        future_forecast = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
        future_dates = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
        forecast_series = pd.Series(future_forecast.flatten(), index=future_dates)

        # Plot actual vs forecast
        fig, ax = plt.subplots(figsize=(12, 5))
        ts[-90:].plot(ax=ax, label="Actual", color='blue')
        forecast_series.plot(ax=ax, label="Forecast", color='red')
        ax.axvline(forecast_series.index[0], linestyle='--', color='gray')
        ax.set_title("LSTM Forecast (Next 30 Days)")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

        # Metrics
        mae = mean_absolute_error(y_test_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        r2 = r2_score(y_test_actual, y_pred)
        mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100
        accuracy = 100 - mape

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("MAPE", f"{mape:.2f}%")
        col3.metric("RMSE", f"{rmse:.2f}")
        col4.metric("R²", f"{r2:.2f}")
        col5.metric("Accuracy", f"{accuracy:.2f}%")

    except Exception as e:
        st.error(f" LSTM Error: {e}")
