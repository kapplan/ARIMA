#!/usr/bin/env python
# coding: utf-8
# Başak Kaplan
# Models used: ARIMA

# Standard library imports
import itertools
import warnings

# Third-party imports
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import periodogram
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
    f1_score,
    confusion_matrix,
    median_absolute_error,
)
from sklearn.model_selection import train_test_split

# Local application/library-specific imports
from chow_test import chow_test
from pmdarima import auto_arima

# Suppress warnings
warnings.filterwarnings('ignore')

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

data.index

# Statistics for each column
data.describe()
# Checking for the missing values
data.isna().sum()

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
             xy=(data.index[0], data.iloc[0]),
             xytext=(data.index[0], data.iloc[0] + 5),
             arrowprops=dict(arrowstyle='->'))
plt.annotate(f'{data.index[-1].strftime("%Y-%m-%d")}',
             xy=(data.index[-1], data.iloc[-1]),
             xytext=(data.index[-1], data.iloc[-1] + 5),
             arrowprops=dict(arrowstyle='->'))
# Display the plot
plt.xticks(rotation=45)
plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.show()

# Seasonal plot
data['year'] = [d.year for d in data.index]
data['month'] = [d.strftime('%b') for d in data.index]
years = data['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(12, 10), dpi=80)
for i, y in enumerate(years):
    if i > 0:
        plt.plot('month', 'Rate', data=data.loc[data.year == y, :], color=mycolors[i], label=y)
        plt.text(data.loc[data.year == y, :].shape[0] - .9, data.loc[data.year == y, 'Rate'][-1:].values[0], y,
                 fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(min(data['Rate']) - 1, max(data['Rate']) + 1), ylabel='$Inflation$',
              xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Inflation Time Series", fontsize=20)
plt.show()

data = data.drop(['year', 'month'], axis=1)

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
    print('====Augmented Dickey-Fuller Test Results ====\n')
    print('ADF Statistic: {:.6f}'.format(result[0]))
    print('p-value: {:.6f}'.format(result[1]))
    print('# Lags used: {}'.format(result[2]))
    print('Number of observations: {}'.format(result[3]))
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}'.format(key, value))

    critical_value = result[4]['5%']
    if (result[1] <= 0.05) and (result[0] < critical_value):
        print('\nReject the null hypothesis. Data has no unit root and is stationary.')
    else:
        print('\nWeak evidence against null hypothesis, time series has a unit root and is non-stationary.')
    return


adf_test(data)

# KPSS Test
# Perform the KPSS test on the 'Rate' column
result = kpss(data['Rate'])

# Print the KPSS test results
print('======= Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test Results =======\n')
print("KPSS Test Statistic:", result[0])
print("p-value:", result[1])
print("# Lags used: ", result[2])
print("Critical Values:")
for key, value in result[3].items():
    print(f"\t{key}: {value}")

critical_value = result[3]['5%']
if result[1] <= 0.05 and result[0] < critical_value:
    print('\nWeak evidence against the null hypothesis, reject the null hypothesis\
            Data has a unit root and is non-stationary.')
else:
    print('\nStrong evidence against null hypothesis, data is stationary.')

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

freq, power = periodogram(data['Rate'])

# Plot the frequency vs. power periodogram for cyclic behavior
plt.plot(freq, power)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Frequency vs. Power Periodogram')
plt.show()

"""" not neccessary since the data is stationary 
from pandas.plotting import register_matplotlib_converters

def transformation(series):
    register_matplotlib_converters()

    fig = plt.figure(figsize=(16, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Transformed Series')
    ax1.plot(series)
    ax1.plot(series.rolling(window=12).mean(), color='crimson')
    ax1.plot(series.rolling(window=12).std(), color='black')

    # Partial Autocorrelation Plot
    ax2 = fig.add_subplot(1, 3, 2)
    plot_acf(series.dropna(), ax=ax2, lags=50, title='Autocorrelation')
    # plot 95% confidence intervals
    plt.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='gray')
    plt.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='gray')
    plt.xlabel('lags')

    # Partial Autocorrelation Plot
    ax3 = fig.add_subplot(1, 3, 3)
    plot_acf(series.dropna(), ax=ax3, lags=50, title="Differenced")
    plt.axhline(y=-1.96 / np.sqrt(len(series)), linestyle='--', color='black')
    plt.axhline(y=1.96 / np.sqrt(len(series)), linestyle='--', color='black')
    plt.xlabel('lags')
    plt.show()

    # ADF Test
    result = adfuller(series.dropna(), regression='c', autolag='AIC')
    critical_value = result[4]['5%']
    if (result[1] <= 0.05) and (result[0] < critical_value):
        print('P value = {:.6f}, series is stationary.'.format(result[1]))
    else:
        print('P value = {:.6f}, series is not stationary.'.format(result[1]))

transformation(data.diff()) """

# creating our ARIMA Model
model = ARIMA(data, order=(2, 0, 2)).fit()
fig, ax = plt.subplots()
ax = data.loc['2018':].plot(ax=ax)
plot_predict(model, '2020', '2025', ax=ax)
plt.show()

# finding the best p d and q values for the arima model
train_data = data[1:len(data) - 12]
test_data = data[len(data) - 12:]

# Define the range of p, d, and q values for ARIMA parameters
p = d = q = range(0, 3)

# Generate all possible combinations of p, d, and q
pdq = list(itertools.product(p, d, q))

# Initialize variables to store the best parameters and minimum AIC
best_aic = float('inf')
best_params = None

# Loop through all combinations of parameters
for param in pdq:
    try:
        mod = sm.tsa.ARIMA(data['Rate'],
                           order=param,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
        results = mod.fit()

        # Check if the current AIC is the best so far
        if results.aic < best_aic:
            best_aic = results.aic
            best_params = param

        print(f'ARIMA{param} - AIC:{results.aic}')
    except Exception as e:
        print(f'Error for ARIMA{param}: {str(e)}')
        continue

# Print the best AIC and corresponding parameters
print(f'Best AIC: {best_aic}')
print(f'Best Parameters: {best_params}')

# In[124]:
model = ARIMA(data['Rate'], order=best_params)
model_fit = model.fit()
print(model_fit.summary())

# Forecasting for the remainder of 2023
prediction = model_fit.forecast(steps=7)
for month in range(10, 13):
    year = 2023
    predicted_inflation = prediction[(prediction.index.year == year) & (prediction.index.month == month)].values[0]
    print(
        f"Poland's predicted inflation rate for {pd.Timestamp(year, month, 1).strftime('%B %Y')} is: {predicted_inflation}%")
# Loop through January and February of 2024
for month in range(1, 3):
    year = 2024
    predicted_inflation = prediction[(prediction.index.year == year) & (prediction.index.month == month)].values[0]
    print(
        f"Poland's predicted inflation rate for {pd.Timestamp(year, month, 1).strftime('%B %Y')} is: {predicted_inflation}%")

# Prediction and Confidence Intervals
pred = model_fit.get_prediction(start=pd.to_datetime('2020-01-01'), dynamic=False)
pred_ci = pred.conf_int()

# Model Evaluation
# Calculate RMSE and MSE
rmse = np.sqrt(mean_squared_error(data['Rate'], results.fittedvalues))
y_forecasted = pred.predicted_mean
y_truth = data['2022-01-01':]['Rate']
mse = ((y_forecasted - y_truth) ** 2).mean()

# Print Results
print(f'Root Mean Square Error: {rmse:.2f}')
print(f'The Mean Squared Error of our forecasts is {round(mse, 2)}')

# Plot the results
fig, ax = plt.subplots(figsize=(14, 7))
data['Rate'].plot(ax=ax, label='Actual')
results.fittedvalues.plot(ax=ax, color='red', label='Fitted')
ax.set_title('Inflation Rates in Poland: Actual vs Fitted')
ax.legend(loc='best')
plt.show()

# Plot residuals
residuals = data['Rate'] - results.fittedvalues
plt.figure(figsize=(14, 7))
plt.plot(residuals.index, residuals, label='Residuals')
plt.title('Residuals of the ARIMA Model')
plt.legend(loc='best')
plt.show()

# Display residuals details
residuals = pd.DataFrame(results.resid)
residuals.plot(title="Residual Errors Over Time")
plt.show()
residuals.plot(kind='kde', title="Density of Residual Errors")
plt.show()
print(residuals.describe())

# Plot the Results
fig, ax = plt.subplots(figsize=(14, 7))
data['2020':].plot(ax=ax, label='Observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Inflation Rate')
ax.legend()
plt.show()

# Plot actual vs. fitted values
fig, ax = plt.subplots(figsize=(14, 7))
data['Rate'].plot(ax=ax, label='Actual')
results.fittedvalues.plot(ax=ax, color='red', label='Fitted')
ax.set_title('Inflation Rates in Poland: Actual vs Fitted')
ax.legend(loc='best')
plt.show()

# Plot residuals
fig, ax = plt.subplots(figsize=(14, 7))
residuals = data['Rate'] - results.fittedvalues
residuals.plot(ax=ax, label='Residuals')
ax.set_title('Residuals of the ARIMA Model')
ax.legend(loc='best')
plt.show()

# Residual error details
fig, ax = plt.subplots(figsize=(14, 7))
residual_errors = pd.DataFrame(results.resid)
residual_errors.plot(ax=ax, title="Residual Errors Over Time")
plt.show()

fig, ax = plt.subplots(figsize=(14, 7))
residual_errors.plot(kind='kde', ax=ax, title="Density of Residual Errors")
plt.show()

# Model Evaluation
# Fit the ARIMA model with the best parameters
model = ARIMA(data['Rate'], order=(2, 0, 2))
results = model.fit()

# Plot actual vs. fitted values
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Rate'], label='Actual')
plt.plot(data.index, results.fittedvalues, color='red', label='Fitted')
plt.title('Inflation Rates in Poland: Actual vs Fitted')
plt.legend(loc='best')
plt.show()

# Calculate residuals
residuals = data['Rate'] - results.fittedvalues

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals.index, residuals, label='Residuals')
plt.title('Residuals of the ARIMA Model')
plt.legend(loc='best')
plt.show()

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# Find the date of the maximum residuals
residuals = data['Rate'] - results.fittedvalues
max_residual_date = residuals.nlargest(1).idxmin()
print(f"The date of the maximum residual is: {max_residual_date}")
# Find the date of the second maximum residual
second_max_residual_date = residuals.nlargest(2).idxmin()
print(f"The date of the second maximum residual is: {second_max_residual_date}")  # 2022-03-01

second_max_residual_date = residuals.nlargest(7).idxmin()
print(f"The date of the second maximum residual is: {second_max_residual_date}")  # 2023-02-01

# Calculate RMSE (Root Mean Square Error)
rmse = np.sqrt(mean_squared_error(data['Rate'], results.fittedvalues))
print(f'Root Mean Square Error: {rmse:.2f}')

# Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
y_forecasted = pred.predicted_mean
y_truth = data['2022-01-01':]['Rate']
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 2)))
