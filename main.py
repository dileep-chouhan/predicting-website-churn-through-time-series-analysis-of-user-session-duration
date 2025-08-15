import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# --- 1. Synthetic Data Generation ---
# Generate synthetic data for user session durations
np.random.seed(42)  # for reproducibility
dates = pd.date_range(start='2023-01-01', periods=365)
session_durations = np.random.randint(1, 60, size=365) + np.sin(np.linspace(0, 2 * np.pi, 365)) * 15 #add seasonality
session_durations[200:250] -= 10 # Simulate a drop in session duration (potential churn)
session_durations = np.maximum(session_durations, 1) # Ensure durations are positive
data = {'Date': dates, 'SessionDuration': session_durations}
df = pd.DataFrame(data)
df = df.set_index('Date')
# --- 2. Data Cleaning & Analysis ---
# (In this synthetic data, cleaning is minimal.  Real-world data would need more robust cleaning.)
# Decompose the time series to identify trend, seasonality, and residuals
decomposition = seasonal_decompose(df['SessionDuration'], model='additive', period=30) #period is approximately monthly
# --- 3. Visualization ---
# Plot the decomposed time series
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df['SessionDuration'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(decomposition.resid, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
output_filename = 'time_series_decomposition.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 4. Forecasting (Simple Exponential Smoothing) ---
# Fit a simple exponential smoothing model
model = SimpleExpSmoothing(df['SessionDuration'])
model_fit = model.fit()
# Forecast the next 30 days
forecast = model_fit.forecast(30)
# --- 5. Visualization of Forecast ---
plt.figure(figsize=(10,6))
plt.plot(df['SessionDuration'], label='Observed')
plt.plot(forecast, label='Forecast')
plt.legend(loc='best')
plt.title('Session Duration Forecast')
plt.xlabel('Date')
plt.ylabel('Session Duration')
output_filename2 = 'session_duration_forecast.png'
plt.savefig(output_filename2)
print(f"Plot saved to {output_filename2}")
print("Note: This is a simplified example.  Real-world churn prediction would involve more sophisticated models and feature engineering.")