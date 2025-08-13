import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import logging
from datetime import timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Step 1: Define AQI calculation function based on India's CPCB breakpoints
def calculate_aqi_pm25(pm25):
    """Calculate AQI based on PM2.5 concentration (µg/m³) using India's CPCB breakpoints."""
    if pd.isna(pm25):
        return np.nan
    if pm25 <= 30:
        aqi = (50 / 30) * pm25
    elif pm25 <= 60:
        aqi = 50 + (50 / 30) * (pm25 - 30)
    elif pm25 <= 90:
        aqi = 100 + (100 / 30) * (pm25 - 60)
    elif pm25 <= 120:
        aqi = 200 + (100 / 30) * (pm25 - 90)
    elif pm25 <= 250:
        aqi = 300 + (100 / 130) * (pm25 - 120)
    else:
        aqi = 400 + (100 / (250)) * (pm25 - 250)
        aqi = min(aqi, 500)  # Cap at 500
    return round(aqi)

def preprocess_data(file_path, station_id):
    logger.info(f"Loading dataset for {station_id}...")
    data = pd.read_csv(file_path, parse_dates=['From Date'])
    data = data.rename(columns={'From Date': 'ds', 'PM2.5 (ug/m3)': 'PM2.5'})

    # Calculate AQI from PM2.5
    data['AQI'] = data['PM2.5'].apply(calculate_aqi_pm25)

    data = data[['ds', 'AQI']].copy()
    data = data.set_index('ds')
    data = data.sort_index()

    # Create a complete hourly index and interpolate missing AQI values
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='H')
    data = data.reindex(full_index)
    data['AQI'] = data['AQI'].interpolate(method='linear')

    # Aggregate to daily data
    data_daily = data.resample('D').mean().interpolate(method='linear')
    return data_daily, station_id

# Step 3: Function to fit SARIMA and forecast
def fit_and_forecast(data_daily, station_id):

    adf_result = adfuller(data_daily['AQI'].dropna())
    logger.info(f"{station_id} ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")

    # Plot AQI time series
    plt.figure(figsize=(12, 6))
    plt.plot(data_daily.index, data_daily['AQI'], label=f'Daily AQI ({station_id})')
    plt.title(f'Daily AQI ({station_id})')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.legend()
    plt.savefig(f'aqi_trend_{station_id}.png')
    plt.close()


    # Fit SARIMA model
    logger.info(f"Fitting SARIMA model for {station_id}...")
    model = SARIMAX(
        data_daily['AQI'],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 365),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    logger.info(f"{station_id} Model summary:\n{results.summary()}")

    # Generate forecasts
    forecast_horizons = {
        '1_month': 30,
        '6_months': 180,
        '1_year': 365,
        '2_years': 730
    }
    forecasts = {}

    for horizon_name, days in forecast_horizons.items():
        logger.info(f"Generating {horizon_name} forecast for {station_id}...")
        forecast = results.get_forecast(steps=days)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int(alpha=0.05)
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(start=data_daily.index[-1] + timedelta(days=1), periods=days, freq='D'),
            'AQI_Forecast': forecast_mean,
            'Lower_CI': forecast_ci.iloc[:, 0],
            'Upper_CI': forecast_ci.iloc[:, 1]
        })
        forecasts[horizon_name] = forecast_df

        forecast_df.to_csv(f'aqi_forecast_{station_id}_{horizon_name}.csv', index=False)

        # Plot forecast
        plt.figure(figsize=(12, 6))
        plt.plot(data_daily.index, data_daily['AQI'], label='Historical AQI')
        plt.plot(forecast_df['Date'], forecast_df['AQI_Forecast'], label=f'Forecast ({horizon_name})')
        plt.fill_between(
            forecast_df['Date'],
            forecast_df['Lower_CI'],
            forecast_df['Upper_CI'],
            alpha=0.2,
            label='95% Confidence Interval'
        )
        plt.title(f'AQI Forecast ({station_id} - {horizon_name})')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.legend()
        plt.savefig(f'aqi_forecast_{station_id}_{horizon_name}.png')
        plt.close()

    return forecasts, mae, rmse

# Step 4: Process multiple stations
stations = [
    ('AP001.csv', 'AP001'),
    ('AP003.csv', 'AP003')
]
all_forecasts = {}
validation_metrics = {}

for file_path, station_id in stations:
    logger.info(f"Processing {station_id}...")
    data_daily, station_id = preprocess_data(file_path, station_id)
    forecasts, mae, rmse = fit_and_forecast(data_daily, station_id)
    all_forecasts[station_id] = forecasts
    validation_metrics[station_id] = {'MAE': mae, 'RMSE': rmse}

# Step 5: Output summary
print("\n=== AQI Forecast Summary ===")
for station_id, forecasts in all_forecasts.items():
    print(f"\nStation {station_id}:")
    for horizon_name, forecast in forecasts.items():
        print(f"\n{horizon_name.replace('_', ' ').title()}:")
        print(forecast[['Date', 'AQI_Forecast', 'Lower_CI', 'Upper_CI']].head(5).to_string(index=False))
    print(f"\nValidation Metrics for {station_id}:")
    print(f"MAE: {validation_metrics[station_id]['MAE']:.2f}")
    print(f"RMSE: {validation_metrics[station_id]['RMSE']:.2f}")
print("\nForecasts saved as CSV files and plots saved as PNG files.")
