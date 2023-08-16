#!/usr/bin/env python
# coding: utf-8
# Team name: Tiger
# Team members: Başak Kaplan

# Model used: ARIMA and Random Forest Regessor. Random Forest Regressor have been chosen as the final model due to lower MSE and MAE.
# Preperation: I have eliminated unneccessary columns and just left the date and the inflation rate values. I have worked with these variables throughout the whole project. I split my dataset into training and test sets. The training set is used to train themodels, and the test set is used to evaluate its performance.
# Training: The Random Forest Regressor is trained on the training data, where it builds multiple decision trees. Each decision tree is trained on a random subset of the features and data samples from the training set. This randomness helps to introduce diversity among the trees.
# Prediction: Once the model is trained, I created a new variable for May 2023 and pass it as input to the trained Random Forest Regressor model. The model then predicts the inflation rate for May based on the learned patterns and relationships from the training data.
# Evaluation: After obtaining the predictions, I compared them with the actual values to assess the performance of the Random Forest Regressor model. Mean Squared Error (MSE) and Mean Absolute Error (MAE) has been calculated to measure the accuracy of the predictions. Then, I decided to drop the ARIMA model.

import warnings

import matplotlib.pyplot as plt
import numpy as np
# Importing the necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

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

# In[119]: Converting the data column into datetime format
data['Date'] = pd.to_datetime(data['Date'])
# Statistics for each column
data.describe()
# In[120]: Checking for the missing values
data.isna().sum()

# In[123]: Exploratory Data Analysis
data['Date'] = pd.to_numeric(data['Date'])


def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):
    """
        series - dataframe with timeseries
        window - rolling window size
        plot_intervals - show confidence intervals
        plot_anomalies - show anomalies
    """
    rolling_mean = series.rolling(window=12, min_periods=1).mean()
    std_rolling = series.rolling(window=12).std()

    plt.figure(figsize=(15, 5))
    plt.title("Moving average\n window size = {}".format(window))
    plt.plot(rolling_mean, "g", label="Rolling mean trend")

    # Plot confidence intervals for smoothed values
    if plot_intervals:
        mae = mean_absolute_error(series[window:], rolling_mean[window:])
        deviation = np.std(series[window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")

        # Having the intervals, find abnormal values
        if plot_anomalies:
            anomalies = pd.DataFrame(index=series.index, columns=series.columns)
            anomalies[series < lower_bond] = series[series < lower_bond]
            anomalies[series > upper_bond] = series[series > upper_bond]
            plt.plot(anomalies, "ro", markersize=10)

    plt.plot(series[window:], label="Actual values")
    plt.legend(loc="upper left")
    plt.grid(True)


plotMovingAverage(data, window=1, plot_intervals=True, plot_anomalies=True)
plt.show()

# Rolling mean and standard deviation
mean_rolling = data['Rate'].rolling(window=12).mean()
std_rolling = data['Rate'].rolling(window=12).std()

# Plot inflation rates
plt.figure(figsize=(12, 5))
data['Rate'].plot(label='Original')
mean_rolling.plot(color='crimson', label='Rolling Mean')
std_rolling.plot(color='black', label='Rolling Std')
plt.title('Inflation Rates in Poland')
plt.grid(axis='y', alpha=0.5)
plt.legend(loc='best')
plt.show()

# Seasonal Decomposition
# Set the frequency of the index to monthly start
data.index.freq = 'MS'
print(data)
# Seasonal decomposition
decomp = seasonal_decompose(data['Rate'], model='additive')
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

# In[123]: Stationary Check - Augmented Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller

# ADF Test
result = adfuller(data['Rate'])
print('======= Augmented Dickey-Fuller Test Results =======\n')
print("ADF Statistic:", result[0])
print("p-value:", result[1])
print("# Lags used: ", result[2])
print("No of observations: ", result[3])
print("Critical Values:")
for key, value in result[4].items():
    print(f"\t{key}: {value}")

critical_value = result[4]['5%']
if (result[1] <= 0.05) and (result[0] < critical_value):
    print('\nStrong evidence against the null hypothesis, reject the null hypothesis.\
            Data has no unit root and is stationary.')
else:
    print('\nWeak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary.')

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
    print('\nWeak evidence against the null hypothesis, reject the null hypothesis.\
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


transformation(data.diff())

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

model = ARIMA(data['Rate'], order=(1, 0, 1))
model_fit = model.fit()

# In[125]: Forecassting for the whole year
prediction = model_fit.forecast(steps=12)

# In[126]:
# Loop through each month and year in 2023
for month in range(8, 13):
    year = 2023
    predicted_inflation = prediction[(prediction.index.year == year) & (prediction.index.month == month)].values[0]
    print(
        f"Poland's predicted inflation rate for {pd.Timestamp(year, month, 1).strftime('%B %Y')} is: {predicted_inflation}%")


# In[127]: # SARIMAX model
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
from datetime import datetime

X = data["Date"].values.reshape(-1, 1)
y = data["Value"]

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y)

# new variable for May 2023
may_2023 = pd.DataFrame({'Date': [datetime(2023, 5, 1)]})  # Adjust the year and month accordingly

# prediction for may
predicted_inflation_may_2023 = rf_regressor.predict(may_2023)

# the difference from the previous month (April 2023)
previous_month_inflation = dt[dt["Date"] == datetime(2022, 5, 1)]["Value"].values[0]
difference = predicted_inflation_may_2023 - previous_month_inflation

print("Poland's Predicted Inflation Rate for May 2023:", predicted_inflation_may_2023)
print("Difference from the previous year same month (May 2022):", difference)

# In[130]:
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Assuming you have a DataFrame 'data' with columns 'Date' and 'Value'
# Split the data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(dt.drop('Value', axis=1), dt['Value'], test_size=0.2,
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

# In[131]:
from sklearn.metrics import mean_squared_error, mean_absolute_error


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# Assuming you have the actual values stored in 'actual_values' and the predictions from each model
# stored in 'arima_predictions' and 'rf_predictions' respectively.

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

#
