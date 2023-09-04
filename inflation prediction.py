#!/usr/bin/env python
# coding: utf-8
# Team members: Başak Kaplan

# Models used: ARIMA and SARIMA, prophet.

# Importing the necessary libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import f1_score, confusion_matrix
from chow_test import chow_test

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import warnings

warnings.filterwarnings('ignore')

# Import data
file_path = '/Users/apple/Downloads/prc_hicp_manr__custom_7158942_linear.csv'
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
decomp = seasonal_decompose(data['Rate'], model='Multiplicable')
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


# Stationary Check - Augmented Dickey-Fuller Test
# ADF statistical test
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
from statsmodels.tsa.stattools import kpss

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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

from scipy.signal import periodogram
freq, power = periodogram(data['Rate'])

# Plot the frequency vs. power periodogram
plt.plot(freq, power)
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('Frequency vs. Power Periodogram')
plt.show()

"""" not neccessary to transform since the data is stationary 
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

# finding the best p d and q values for the arima model
train_data = data[1:len(data) - 12]
test_data = data[len(data) - 12:]

p_values = [0, 1]
d_values = range(0, 2)
q_values = range(0, 2)

for p in p_values:
    for d in d_values:
        for q in q_values:
            order = (p, d, q)
            warnings.filterwarnings("ignore")
            model = ARIMA(train_data['Rate'], order=order).fit()
            predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
            error = mean_squared_error(test_data['Rate'], predictions)
            print('ARIMA%s MSE=%.3f' % (order, error))

# In[124]:

model = ARIMA(data['Rate'], order=(1, 1, 1))
model_fit = model.fit()

# In[125]: Forecasting for the whole year
prediction = model_fit.forecast(steps=12)

# In[126]:
# Loop through each month and year in 2023
for month in range(8, 9):
    year = 2023
    predicted_inflation = prediction[(prediction.index.year == year) & (prediction.index.month == month)].values[0]
    print(
        f"Poland's predicted inflation rate for {pd.Timestamp(year, month, 1).strftime('%B %Y')} is: {predicted_inflation}%")


# SARIMAX model
# find best orders and evaluate each combination for SARIMAX model
# series: must be a pandas dataframe
def find_optimal_orders(series, verbose=True):
    # filter out harmless warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
    warnings.simplefilter('ignore', (ConvergenceWarning, ValueWarning))

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import time

    # start timer
    start_time = time.time()

    ######### list of possible combinations
    order_list = []
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                order_list.append((p, d, q))

    ######### initialize variables
    m = 12
    trend_pdq = order_list
    seasonal_pdq = [(x[0], x[1], x[2], m) for x in order_list]
    min_aic = float('inf')
    best_params = {'order': None, 'seasonal_order': None}

    ######### loop through every possible configuration and print results
    print('Expected Fits: {}'.format(len(trend_pdq) * len(trend_pdq)))
    print('========== SARIMAX Results ==========\n')
    count = 0
    for param in trend_pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(endog=series, order=param, seasonal_order=param_seasonal, freq='M', exog=None,
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()

                if verbose:
                    count += 1
                    print('{}. SARIMAX{}{}[{}],\tAIC = {:.6f},\tBIC = {:.6f}'.format(count, param,
                                                                                     param_seasonal[:-1], m,
                                                                                     model_fit.aic, model_fit.bic))

                if model_fit.aic < min_aic:
                    min_aic = model_fit.aic
                    best_params['order'] = param
                    best_params['seasonal_order'] = param_seasonal
                    line = count
            except:
                print('Error while fitting model')
                continue
    print('\nBest order: {}{}[{}] with AIC = {:.6f} at line {}'.format(best_params['order'],
                                                                       best_params['seasonal_order'][:-1], m, min_aic,
                                                                       line))

    # stop timer and display execution time
    diff = time.time() - start_time
    print('\n(Total time of execution: {:.0f} min {:.2f} s)'.format(diff % 3600 // 60, diff % 60))


# display results
find_optimal_orders(data['Rate'], verbose=True)

# In[129]:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from datetime import datetime

data.reset_index(inplace=True)
X = data['Date'].values.reshape(-1, 1)
y = data['Rate']

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)

# new variable for May 2023
august_2023 = pd.DataFrame({'Date': [datetime(2023, 8, 1)]})  # Adjust the year and month accordingly

# prediction for may
predicted_inflation_august_2023 = rf_regressor.predict(august_2023)

# the difference from the previous month (April 2023)
previous_month_inflation = data[data["Date"] == datetime(2022, 8, 1)]["Rate"].values[0]
difference = predicted_inflation_august_2023 - previous_month_inflation

print("Poland's Predicted Inflation Rate for August 2023:", predicted_inflation_august_2023)
print("Difference from the previous year same month (August 2022):", difference)

# In[130]:
X_train, X_test, y_train, y_test = train_test_split(data.drop('Rate', axis=1), data['Rate'], test_size=0.2,
                                                    random_state=42)

# Create and train the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_regressor.predict(X_test)

# Get the actual values from the test set
actual_values = y_test

# Print the actual values and predictions
print("Actual Values:")
print(actual_values)
print("\nRandom Forest Regressor Predictions:")
print(rf_predictions)


#MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MSE
arima_mse = mean_squared_error(actual_values, arima_predictions)
rf_mse = mean_squared_error(actual_values, rf_predictions)

# Calculate MAE
arima_mae = mean_absolute_error(actual_values, arima_predictions)
rf_mae = mean_absolute_error(actual_values, rf_predictions)

# Print the results
print("ARIMA Model:")
print("MSE:", arima_mse)
print("MAE:", arima_mae)
print("\nRandom Forest Regressor Model:")
print("MSE:", rf_mse)
print("MAE:", rf_mae)

# Compare the models
if arima_mse < rf_mse:
    print("\nThe ARIMA model has a lower MSE and may be preferred.")
elif arima_mse > rf_mse:
    print("\nThe Random Forest Regressor model has a lower MSE and may be preferred.")
else:
    print("\nThe ARIMA and Random Forest Regressor models have the same MSE.")

if arima_mae < rf_mae:
    print("The ARIMA model has a lower MAE and may be preferred.")
elif arima_mae > rf_mae:
    print("The Random Forest Regressor model has a lower MAE and may be preferred.")
else:
    print("The ARIMA and Random Forest Regressor models have the same MAE.")

# Prophet
from prophet import Prophet

# Rename columns to fit Prophet's naming convention
data.reset_index(inplace=True)
data.rename(columns={'Date': 'ds', 'Rate': 'y'}, inplace=True)

# Initialize and fit the Prophet model
model = Prophet()
model.fit(data)

# Create a dataframe for future dates (August 2023)
future = pd.DataFrame({'ds': pd.date_range(start='2023-08-01', end='2023-12-31', freq='M')})

# Make predictions
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Inflation Rate Forecast for August 2023')
plt.xlabel('Date')
plt.ylabel('Inflation Rate')
plt.show()

# Step 3: Print the predicted rates for each month and their values
predicted_values = forecast[['ds', 'yhat']].tail()  # Get the last 5 predictions
print("Predicted Inflation Rates for August to December 2023:")
print(predicted_values)

from pylab import rcParams
import itertools

# SARIMAX
# Import data
file_path = '/Users/apple/Downloads/prc_hicp_manr__custom_7158942_linear.csv'
data = pd.read_csv(file_path)
data = data.drop(['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'geo', 'OBS_FLAG', 'coicop'], axis=1). \
    rename(columns={'TIME_PERIOD': 'Date', 'OBS_VALUE': 'Rate'})

# Converting the data column into datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

y = data['Rate'].resample('MS').mean()
data.index

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

import statsmodels.api as sm

best_aic = float("inf")
best_order = None
best_seasonal_order = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            aic = results.aic
            if aic < best_aic:
                best_aic = aic
                best_order = param
                best_seasonal_order = param_seasonal
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
        except:
            continue

print('Best ARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2020-07-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2018':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Inflation')
plt.legend()
plt.show()

# Model Evaluation
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=36)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='Observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Inflation')
plt.legend()
plt.show()

# Step 3: Print the predicted rates for each month and their values
predicted_values = forecast[['ds', 'yhat']].tail()  # Get the last 5 predictions
print("Predicted Inflation Rates for August to December 2023:")
print(predicted_values)
