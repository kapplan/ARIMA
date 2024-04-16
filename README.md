# Time Series Forecasting and Analysis using ARIMA (Predicting Poland's Inflation)

This project is dedicated to forecasting inflation rates for Poland in 2024 using advanced time series analysis
techniques in Python. The core methodologies employed include ARIMA, one-step ahead forecasting, and recursive
forecasting, alongside a comprehensive analysis of stationarity and structural breaks within the dataset.

## Detailed Analysis and Project Walkthrough

For a comprehensive walkthrough of the entire project process, including in-depth analyses and step-by-step
explanations, please visit my Medium page. You can find detailed articles and discussions about the methodologies,
results, and insights gained during this project:
[Time Traveling with Data: Forecasting Inflation for 2024](https://medium.com/@kapplan)

## Findings

- **The periodogram indicated a seasonal component at a frequency of 0.0003, corresponding to a peak
  power of 31741.7432, signaling seasonal patterns.**
- **After the first differencing, the stationarity of the time series was confirmed with both the Augmented
  Dickey-Fuller (ADF) and the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests.**
- **Structural breaks were identified in the time series, which may correspond to significant external events such as
  COVID-19 pandemic, Russian invasion of Ukraine, Poland's new government after general elections of 2023, affecting
  inflation rates.**
- **The model was fitted with ARIMA(2, 1, 5) parameters, chosen based on the lowest AIC and BIC values, which
  suggested a moderate fit.**
- **Forecasting for the next 5 months of 2024, the model predicts inflation rates ranging from 3.29% in March to 1.89%
  in July, highlighting a decreasing trend in inflation.**
- **The residuals analysis revealed that while most forecast errors were minimal, there were instances of significant
  overestimation or underestimation, suggesting the model's occasional sensitivity to outliers or particular economic
  events.**

### Key Features

- Time Series Decomposition
- Stationarity Testing
- Predictive Modeling
- Visualization
- Structural Break Analysis
- Recursive Forecast
- Model Evaluation

Time series data consists of a sequence of data points collected at regular time intervals. This type of data is
essential in various industries, including finance, economics, and weather forecasting. Analyzing time series data
allows us to uncover valuable patterns, trends, and correlations that enable informed decision-making and predictions.

### Objective

- Understand the characteristics of the inflation rate data. Identify trends, seasonality, and any patterns that might
  influence the inflation rate.
- Build a model that can predict the future values of inflation rate in Poland for the first half of 2024 using HICP
  data.

### Problem definition

Rising inflation reduces consumers' purchasing power. It can erode consumer confidence in spending, leading to an
economic slowdown. High inflation can lead to uncertainty about future costs and revenues, impacting investment
decisions. Inflation often leads to higher costs for raw materials, labor, and energy, reducing profit margins.
Additionally, High inflation can lead to currency depreciation, affecting international trade and investments.
By addressing these potential risks and implementing appropriate mitigation strategies, businesses and governments can
better navigate the challenges associated with increasing inflation, foster resilience, and minimize negative impacts.

### Dataset

The dataset consists of 326 monthly observations of Poland's HICP.
Data is collected from Eurostat. Eurostat is the statistical office of the European Union, responsible for publishing
high-quality Europe-wide statistics and
indicators: https://ec.europa.eu/eurostat/databrowser/view/PRC_HICP_MANR__custom_7158942/default/table?lang=en

### Installation

Install the project dependencies via pip with the following command:

```pip install numpy pandas matplotlib scipy statsmodels scikit-learn seaborn ruptures pmdarima```

### Usage

Follow the steps below to get started with the application:

1. **Data Preparation**: Load your dataset into a pandas DataFrame and ensure it is properly formatted with the time
   series data indexed by date.

2. **Model Selection**: Choose between ARIMA and SARIMAX based on the data's characteristics.

3. **Forecasting**: Generate predictions using the chosen model and evaluate its performance.

4. **Analysis**: Conduct tests for stationarity, visualize the data, and check for structural breaks.

Example of running an ARIMA model:

```
from statsmodels.tsa.arima.model import ARIMA 

# Example ARIMA model fitting
model = ARIMA(data['Rate'], order=(2, 1, 5))
model_fit = model.fit()
print(model_fit.summary())
```

### Contributing

Contributions are welcome! Please fork the repository and submit pull requests with your proposed changes :)
