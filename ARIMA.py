# Standard library imports
import itertools
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Third-party imports for data handling
import numpy as np
import pandas as pd

# Third-party imports for statistical modeling
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from pmdarima import auto_arima
import ruptures as rpt

# Third-party imports for machine learning
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
    f1_score,
    confusion_matrix,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Third-party imports for plotting and visualization
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import pyplot
from pandas.plotting import register_matplotlib_converters
from scipy.stats import pearsonr
from scipy import stats
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
import seaborn as sns
from pylab import rcParams
from sklearn import metrics
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning

# Plot settings
plt.style.use('seaborn')
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.color'] = 'k'

# Import data
file_path = '/Users/apple/Downloads/prc_hicp_manr__custom_7843973_linear.csv'
data = pd.read_csv(file_path)

data.head()
data.tail()

# See the data types and non-missing values
data.info()

# Cleaning and Formatting
data = data.drop(['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'geo', 'OBS_FLAG', 'coicop'], axis=1). \
    rename(columns={'TIME_PERIOD': 'Date', 'OBS_VALUE': 'Rate'})

# Converting the data column into datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.resample('MS').mean()

# Statistics for each column
data.describe()

# Checking for the missing values
data.isna().sum()

# Setting up the figure and axes for a 2x2 grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

# Histogram
ax1.hist(data['Rate'], bins=30, alpha=0.75, color='blue')
ax1.set_title("Histogram of Rates")
ax1.set_xlabel("Rate")
ax1.set_ylabel("Frequency")

# Box Plot
ax2.boxplot(data['Rate'], vert=False)
ax2.set_title("Box Plot of Rates")
ax2.set_xlabel("Rate")

# Q-Q Plot
stats.probplot(data['Rate'], dist="norm", plot=ax3)
ax3.set_title("Q-Q Plot")

# Time Series Plot
data['Date'] = pd.date_range(start='1/1/2020', periods=len(data), freq='D')
ax4.plot(data['Date'], data['Rate'], marker='', linestyle='-', color='blue')
ax4.set_title("Time Series Plot of Rates")
ax4.set_xlabel("Date")
ax4.set_ylabel("Rate")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

# Visualization
# Rolling mean and standard deviation
mean_rolling = data['Rate'].rolling(window=12).mean()
std_rolling = data['Rate'].rolling(window=12).std()

plt.figure(figsize=(12, 5))
plt.plot(data.index, data['Rate'], label='Original')
plt.plot(mean_rolling.index, mean_rolling, color='crimson', label='Rolling Mean')
plt.plot(std_rolling.index, std_rolling, color='black', label='Rolling Std')
plt.title('Inflation Rates in Poland')
plt.grid(which='major', linestyle='--', alpha=0.5)

# Format the x-axis date labels
date_format = mdates.DateFormatter('%Y-%m-%d')
plt.gca().xaxis.set_major_formatter(date_format)
plt.legend(loc='best')

# Get the first and last dates in the dataset
first_date = data.index[0]
last_date = data.index[-1]

# Annotate the starting and ending points of the rolling mean
plt.annotate(f'{data.index[0].strftime("%Y-%m-%d")}',
             xy=(data.index[0], data['Rate'].iloc[0]),
             xytext=(data.index[0], data['Rate'].iloc[0] + 2),
             arrowprops=dict(arrowstyle='->'))

plt.annotate(f'{data.index[-1].strftime("%Y-%m-%d")}',
             xy=(data.index[-1], data['Rate'].iloc[-1]),
             xytext=(data.index[-1], data['Rate'].iloc[-1] + 2),
             arrowprops=dict(arrowstyle='->'))
# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()

# Seasonal Decomposition
# Set the frequency of the index to monthly start
decomp = seasonal_decompose(data['Rate'], model='Adittive')
fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(12, 5))
fig.suptitle('Seasonal Decomposition')

decomp.trend.plot(ax=axes[0], legend=False)
axes[0].set_ylabel('Trend')
decomp.seasonal.plot(ax=axes[1], legend=False)
axes[1].set_ylabel('Seasonal')
decomp.resid.plot(ax=axes[2], legend=False)
axes[2].set_ylabel('Residual')
decomp.observed.plot(ax=axes[3], legend=False)
axes[3].set_ylabel('Original')
plt.show()

df_add = pd.concat([decomp.trend, decomp.seasonal, decomp.resid, decomp.observed], axis=1)
df_add.columns = ['trend', 'seasoanilty', 'residual', 'actual_values']
df_add.head()

# Stationary Check
# ADF Test
def adf_test(series):
    result = adfuller(series, regression='c', autolag='AIC')
    print('====Augmented Dickey-Fuller Test Results====\n')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'# Lags used: {result[2]}')
    print(f'Number of observations: {result[3]}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.6f}')

    # Simplifying the logic for rejecting the null hypothesis
    if result[1] < 0.05:
        print('\nReject the null hypothesis. Data has no unit root and is stationary.')
    else:
        print('\nCannot reject the null hypothesis. Data may have a unit root and be non-stationary.')

adf_test(data['Rate'])

# KPSS Test
# Perform the KPSS test on the 'Rate' column
result = kpss(data['Rate'])

print('======= Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test Results =======\n')
print("KPSS Test Statistic:", result[0])
print("p-value:", result[1])
print("# Lags used: ", result[2])
print("Critical Values:")
for key, value in result[3].items():
    print(f"\t{key}: {value}")

# Interpretation of the KPSS test results
critical_value = result[3]['5%']
if result[0] > critical_value:
    print('\nStrong evidence against the null hypothesis, we reject the null hypothesis. Data is non-stationary.')
else:
    print('\nWeak evidence against rejecting the null hypothesis. Data has no unit root and is stationary.')

# ACF and PACF
f, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
plot_acf(data['Rate'], lags=20, ax=ax[0])
plot_pacf(data['Rate'], lags=20, ax=ax[1], method='ols')

ax[1].annotate('Strong correlation at lag = 1', xy=(1, 0.36), xycoords='data',
               xytext=(0.15, 0.7), textcoords='axes fraction',
               arrowprops=dict(color='red', shrink=0.05, width=1))

ax[1].annotate('Strong correlation at lag = 2', xy=(2.1, -0.5), xycoords='data',
               xytext=(0.25, 0.1), textcoords='axes fraction',
               arrowprops=dict(color='red', shrink=0.05, width=1))
plt.tight_layout()
plt.show()

# Cyclic Behavior Detection: Frequency vs. periodogram
fs = 1 / 12  # The sampling frequency. For monthly data, it would be 1 sample per month.
# Compute the periodogram
frequencies, power = periodogram(data['Rate'], fs=fs)

# Print the first few values
print("Frequency\tPower")
for freq, pwr in zip(frequencies[:10], power[:10]):  # Adjust the slice as needed
    print(f"{freq:.4f}\t\t{pwr:.4f}")

# Find the index of the maximum power
max_power_index = np.argmax(power)
max_frequency = frequencies[max_power_index]
max_power = power[max_power_index]

print(f"Peak Frequency: {max_frequency:.4f}")
print(f"Peak Power: {max_power:.4f}")

# Plot the frequency vs. power periodogram for cyclic behavior
plt.plot(frequencies, power)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Frequency vs. Power Periodogram')
plt.show()

# Function to transform and check stationarity
def transformation(series):
    # Differencing the series
    diff_series = series.diff().dropna()

    # Register the converters for matplotlib
    register_matplotlib_converters()

    # Plot the transformed (differenced) series
    fig = plt.figure(figsize=(16, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Transformed Series')
    ax1.plot(diff_series, label='Differenced Series')
    ax1.plot(diff_series.rolling(window=12).mean(), color='crimson', label='Rolling Mean')
    ax1.plot(diff_series.rolling(window=12).std(), color='black', label='Rolling Std')
    ax1.legend()

    # Autocorrelation Plot
    ax2 = fig.add_subplot(1, 3, 2)
    plot_acf(diff_series, ax=ax2, lags=50, title='Autocorrelation')
    ax2.axhline(y=-1.96 / np.sqrt(len(diff_series)), linestyle='--', color='gray')
    ax2.axhline(y=1.96 / np.sqrt(len(diff_series)), linestyle='--', color='gray')
    ax2.set_xlabel('Lags')

    # Partial Autocorrelation Plot
    ax3 = fig.add_subplot(1, 3, 3)
    plot_pacf(diff_series, ax=ax3, lags=50, title="Partial Autocorrelation")
    ax3.axhline(y=-1.96 / np.sqrt(len(diff_series)), linestyle='--', color='gray')
    ax3.axhline(y=1.96 / np.sqrt(len(diff_series)), linestyle='--', color='gray')
    ax3.set_xlabel('Lags')

    plt.tight_layout()
    plt.show()

    # ADF Test to check Stationarity
    adf_result = adfuller(diff_series, autolag='AIC')
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    for key, value in adf_result[4].items():
        print(f'Critical Value ({key}): {value}')
    if adf_result[1] < 0.05:
        print('Series is stationary according to ADF test.')
    else:
        print('Series is not stationary according to ADF test.')

    # KPSS Test
    kpss_result = kpss(diff_series, nlags='auto')
    print("KPSS Test Statistic:", kpss_result[0])
    print("p-value:", kpss_result[1])
    for key, value in kpss_result[3].items():
        print(f'Critical Value ({key}): {value}')
    if kpss_result[1] < 0.05:
        print('Evidence suggests the series is not stationary according to KPSS test.')
    else:
        print('No evidence against the null hypothesis; the series is stationary according to KPSS test.')

    return diff_series

# Example usage
transformation(data['Rate'])

# Train Test Split for finding the Optimal Paramaters
train_data = data[1:len(data) - 12]
test_data = data[len(data) - 12:]

# Define the range of p, d, and q values for ARIMA parameters
# Since 'd' is already given as 1 we only define ranges for 'p' and 'q'
p = range(0, 3)
q = range(0, 7)
d = 1

# Generate all possible combinations of p, d, and q (with d fixed at 1)
pdq = list(itertools.product(p, [d], q))

# Initialize variables to store the best parameters and minimum AIC
best_aic = float('inf')
best_aic_params = None

# Loop through all combinations of parameters
for param in pdq:
    try:
        mod = sm.tsa.ARIMA(train_data['Rate'],
                           order=param,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        results = mod.fit()

        # Check if the current AIC is the best so far
        if results.aic < best_aic:
            best_aic = results.aic
            best_aic_params = param

        print(f'ARIMA{param} - AIC:{results.aic}')
    except Exception as e:
        print(f'Error for ARIMA{param}: {str(e)}')
        continue

# Print the best AIC and corresponding parameters
print(f'Best AIC: {best_aic}')
print(f'Best AIC Parameters: {best_aic_params}')

# Best BIC
# Only define ranges for 'p' and 'q' based on ACF and PACF
p = range(0, 3)
q = range(0, 7)
d = 1  # d is already determined to be 1

# Generate all possible combinations of p, d, and q (with d fixed at d=1)
pdq = list(itertools.product(p, [d], q))

# Initialize variables to store the best parameters and minimum BIC
best_bic = float('inf')
best_bic_params = None

# Loop through all combinations of parameters
for param in pdq:
    try:
        mod = sm.tsa.ARIMA(train_data['Rate'],
                           order=param,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        results = mod.fit()

        # Check if the current BIC is the best so far
        if results.bic < best_bic:
            best_bic = results.bic
            best_bic_params = param

        print(f'ARIMA{param} - BIC:{results.bic}')
    except Exception as e:
        print(f'Error for ARIMA{param}: {str(e)}')
        continue

# Print the best BIC and corresponding parameters
print(f'Best BIC: {best_bic}')
print(f'Best BIC Parameters: {best_bic_params}')

# Finding Optimal Parameters using Time Series Split
tscv = TimeSeriesSplit(n_splits=5)

best_aic_tss = float('inf')
best_params_tss = None

for param in pdq:
    try:
        aic_values = []
        for train_index, test_index in tscv.split(data):
            train_data_tss, test_data_tss = data.iloc[train_index], data.iloc[test_index]
            mod = sm.tsa.ARIMA(train_data_tss['Rate'], order=param, enforce_stationarity=False,
                               enforce_invertibility=False)
            results = mod.fit()
            aic_values.append(results.aic)

        mean_aic = np.mean(aic_values)
        if mean_aic < best_aic_tss:
            best_aic_tss = mean_aic
            best_params_tss = param

    except Exception as e:
        continue

print(f'Best AIC (TimeSeriesSplit): {best_aic_tss}')
print(f'Best Parameters (TimeSeriesSplit): {best_params_tss}')

# In[124]:
# AIC with train-test split and AIC of Time Series Split returns the same parameters,
# therefore I will test the model performance of the best BIC parameters and best AIC parameters

# Fit ARIMA model with the best parameters found using Train-Test Split (BIC parameters)
model_train_test_split = ARIMA(data['Rate'], order=best_bic_params)
model_fit_train_test_split = model_train_test_split.fit()

# Fit ARIMA model with the best parameters found using TimeSeriesSplit (AIC parameters)
model_time_series_split = ARIMA(data['Rate'], order=best_params_tss)
model_fit_time_series_split = model_time_series_split.fit()

# Forecast for 6 months into the future using both models
forecast_steps = 6

# Forecast using TimeSeriesSplit model
forecast_time_series_split = model_fit_time_series_split.get_forecast(steps=forecast_steps)
forecast_time_series_split_values = forecast_time_series_split.predicted_mean
forecast_time_series_split_ci = forecast_time_series_split.conf_int()

# Function to print and format the model summary and performance metrics
def print_model_performance(model_fit, model_name):
    print(f"Model Performance: {model_name}")
    print("=" * 80)
    print(model_fit.summary())
    print("\nPerformance Metrics:")
    residuals = data['Rate'] - model_fit.fittedvalues
    mse = mean_squared_error(data['Rate'], model_fit.fittedvalues)
    mae = mean_absolute_error(data['Rate'], model_fit.fittedvalues)
    rmse = np.sqrt(mse)
    r2 = r2_score(data['Rate'], model_fit.fittedvalues)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")
    print("=" * 80)


# Print the performance for the Train-Test Split model
print_model_performance(model_fit_train_test_split, "Train-Test Split Model")

# Print the performance for the TimeSeriesSplit model
print_model_performance(model_fit_time_series_split, "TimeSeriesSplit Model")

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Rate'], label='Observed')
plt.plot(forecast_time_series_split_values.index, forecast_time_series_split_values, label='Forecast (TimeSeriesSplit)')
plt.fill_between(forecast_time_series_split_ci.index,
                 forecast_time_series_split_ci.iloc[:, 0],
                 forecast_time_series_split_ci.iloc[:, 1], color='green', alpha=0.2)
plt.legend()
plt.title('Forecast: TimeSeriesSplit')
plt.show()

# Forecasting for 6 months into the future
forecast_steps = 6
prediction_tss = model_fit_time_series_split.forecast(steps=forecast_steps)
for month in range(3, 9):
    year = 2024
    predicted_inflation_tss = \
        prediction_tss[(prediction_tss.index.year == year) & (prediction_tss.index.month == month)].values[0]
    print(
        f"Poland's predicted inflation rate for {pd.Timestamp(year, month, 1).strftime('%B %Y')} is: {predicted_inflation_tss}%")

# Generate predictions and confidence intervals
pred_tss = model_fit_time_series_split.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=False)
pred_ci_tss = pred_tss.conf_int()

# Plot the actual data and predictions
fig, ax = plt.subplots()
data.loc['2018':].plot(ax=ax)  # Actual data
pred_tss.predicted_mean.plot(ax=ax, style='r--')  # Predicted mean
ax.fill_between(pred_ci_tss.index, pred_ci_tss.iloc[:, 0], pred_ci_tss.iloc[:, 1], color='pink',
                alpha=0.3)  # Confidence interval
ax.set_title('Inflation Rates in Poland: Actual vs Predicted')
ax.legend(['Actual', 'Predicted'])
plt.show()

# Plot the actual data, fitted values, and predictions
fig, ax = plt.subplots(figsize=(14, 7))
data['Rate'].plot(ax=ax, label='Actual')
model_fit_time_series_split.fittedvalues.plot(ax=ax, color='red', label='Fitted')

# Predictions
plot_predict(model_fit_time_series_split, start='2020', end='2025', ax=ax, plot_insample=False)
ax.set_title('Inflation Rates in Poland: Actual vs Fitted vs Predicted')
ax.legend(loc='best')
plt.show()

# Calculate Residuals
residuals_tss = data['Rate'] - model_fit_time_series_split.fittedvalues

# Plot residuals from TimeSeriesSplit model
plt.figure(figsize=(14, 7))
plt.plot(residuals_tss.index, residuals_tss, label='Residuals')
plt.title('Residuals of the ARIMA Model')
plt.legend(loc='best')
plt.show()
print(residuals_tss.describe())

# Density of Residuals from TimeSeriesSplit model
fig, ax = plt.subplots(figsize=(14, 7))
residuals_tss.plot(kind='kde', ax=ax, title="Density of Residual Errors ")
plt.show()

# Find the date of the maximum residual from TimeSeriesSplit model
max_residual_date_tss = residuals_tss.nlargest(1).idxmin()
print(f"The date of the maximum residual is: {max_residual_date_tss}")

# Find the date of the second maximum residual from TimeSeriesSplit model
second_max_residual_date_tss = residuals_tss.nlargest(2).idxmin()
print(f"The date of the second maximum residual is: {second_max_residual_date_tss}")

# Recursive Forecast using TimeSeriesSplit model
def recursive_forecast_tss(data, start_date, end_date, forecast_horizon, order):
    """
       Performs recursive forecasting and calculates forecast errors using TimeSeriesSplit model.

       Parameters:
       - data: pd.Series, time series data with datetime index.
       - start_date: str, initial model estimation period end.
       - end_date: str, last date to include in forecasting.
       - forecast_horizon: int, number of steps ahead to forecast.
       - order: tuple, ARIMA model order (p, d, q).

       Returns:
       - dict: Forecast errors for each horizon (ME, MAE, RMSE).
       """
    forecast_errors = {i: [] for i in range(1, forecast_horizon + 1)}
    naive_forecasts = data.shift(1)
    naive_errors = np.abs(data - naive_forecasts)

    for current_end in pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq='M'):
        train_data = data[:current_end]
        model = ARIMA(train_data, order=order)
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=forecast_horizon)
        actual = data[
                 current_end + pd.offsets.MonthBegin(1):current_end + pd.offsets.MonthBegin(1) + pd.offsets.MonthEnd(
                     forecast_horizon)]

        for i, (pred, act) in enumerate(zip(forecast, actual), start=1):
            error = act - pred
            forecast_errors[i].append(error)
    # Print model summary
    print(f"Model summary for training data ending {current_end}:")
    print(model_fit.summary())

    error_metrics = {}
    for horizon in range(1, forecast_horizon + 1):
        errors = forecast_errors[horizon]
        actuals = data[len(data) - len(errors):]  # Ensuring the same length and alignment
        me = np.mean(errors)
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(np.square(errors)))
        mape = np.mean(np.abs(np.array(errors) / np.where(np.array(actuals) != 0, np.array(actuals),
                                                          np.nan))) * 100  # Avoiding division by zero
        mase = mae / np.mean(naive_errors)

        error_metrics[horizon] = {
            'ME': me,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'MASE': mase
        }

    return error_metrics


# Example usage for recursive forecast:
start_date = '2006-12-01'  # End of initial 10-year training period
end_date = '2024-01-01'  # Allows for validation of the last forecast in February 2024

forecast_horizon = 6
order_tss = best_params_tss

errors_tss = recursive_forecast_tss(train_data['Rate'], start_date, end_date, forecast_horizon, order_tss)
for horizon, metrics in errors_tss.items():
    print(f"Forecast Horizon {horizon} months:")
    print(
        f"ME: {metrics['ME']:.4f}, MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAPE:{metrics['MAPE']:.4F}")
