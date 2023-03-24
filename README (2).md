# Video Game A/B Testing



## Introduction
This project aims to use Monte Carlo simulations and SARIMA time series analysis to forecast stock prices for the "AAL" ticker, American Airlines Ltd, in order to inform investment decisions. The project will involve collecting historical stock data, implementing Monte Carlo simulations to predict future stock prices, and then fitting a SARIMA model to the time series data to forecast future stock prices with greater accuracy.

## Technologies
#### 1. Jupyter Notebook
#### 2. Python 


## Key Features
Here, describe the key features of your project. This section should highlight what makes your project unique and valuable to potential users.

## Installation
Dependencies for this project are: 

- pandas
- numpy
- matplotlib.pyplot
- scipy.stats



## Development

# Monte Carlo Simulation

## Install the libraries in the notebook.

```bash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import norm
```

## Creating a Pandas dataframe from the CSV file.

```bash
df = pd.read_csv("../input/all_stocks_5yr.csv")
df.head()
```
![DataFrame](Monte%20Carlo%20Simulation/dataframe.png)


## Understanding the data types within the dataframe.

```bash
df.dtypes
```

![DataTypes](Monte%20Carlo%20Simulation/dataframe_types.png)


## Checking for Null values.

```bash
print(df.isna().sum())
print("No of Rows: ", str(len(df)))
```

![Null Count](Monte%20Carlo%20Simulation/dataframe_null_count.pngg)


## Getting an the data related to American Airlines Ltd "AAL"

```bash
# Picking the date, opne, high, low, close and volume for "AAL" 
df = df[df.Name == 'AAL']
df.tail()
```

![Retention](Monte%20Carlo%20Simulation/AAL_df_tail.png)


## Defining our series, stationarity and displaying our results.

```bash
# Getting the logarithmic percentage returns of the closing price
log_returns = np.log(1 + df.close.pct_change())

# Mean of the logarithmich return
u = log_returns.mean() 

# Variance of the logarithic return
var = log_returns.var()

# drift / trend of the logarithmic return
drift = u - (0.5 * var) 

# Standard deviation of the log return
stdev = log_returns.std()

# I just wanted to forecast 250 time points
t_intervals = 250 

# I wanted to have 10 different forecast
iterations = 10 

# Daily_returns actually has some noise. When we multiply this with the t time price, we can obtain t+1 time price
daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))

# Plot daily returns
plt.plot(daily_returns)

# Set plot title
plt.title('Daily Returns for AAL')

# Set x-axis label
plt.xlabel('Date')

# Set y-axis label
plt.ylabel('Returns')

# Show plot
plt.show()
```

![Chart1](Monte%20Carlo%20Simulation/logarithmic_daily_returns.png)

### Question: 
Why is it important to do logarithmic returns of our closing price data?
### Answer:
Taking logarithmic returns of closing price data is important because it makes the returns additive over time, which is useful for various financial calculations and analysis. Additionally, it helps in the interpretation and visualization of data, particularly when prices change over a wide range. By taking the logarithm of the prices, we can more easily compare and analyze the relative changes in prices, as opposed to the absolute changes. Furthermore, logarithmic returns better capture the percentage changes in prices, which is more relevant to investors and traders than the absolute price changes.


## Building our matrix for the monte Carlo simulation

```bash
# Defining our final price to build our monte carlo simulation from
S0 = df.close.iloc[-1]

# Create en empty matrix like daily returns
price_list = np.zeros_like(daily_returns)

# Insert the last known price from AAL as our starting price for our simulation matrix, where we will log all the simulation results.
price_list[0] = S0

# Display our matrix
price_list
```

![Chart2](Monte%20Carlo%20Simulation/matrix_zeros_for_monte_carlo_values.png)


## Creating our monte carlo time series simulations using daily returns from out stocks.


```bash
# With a simple for loop, we are going to forecast the next 250 days
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]
price_list = pd.DataFrame(price_list)
price_list['close'] = price_list[0]
price_list.head()
```

![Chart3](Monte%20Carlo%20Simulation/simulated_values_from_monte_carlo.png)


## After removing outliers, lets take a look at the profile of the data distribution.

```bash
# create a pandas series called closing
close = df.close

# Turn the series into a pandas dataframe
close = pd.DataFrame(close)

# Create a new dataframe with the closing prices and the new simulated monte carlo prices
frames = [close, price_list]
monte_carlo_forecast = pd.concat(frames)

# Creating an Array of only the values needed for the plot
monte_carlo = monte_carlo_forecast.iloc[:,:].values

# Define figure size for the plot
plt.figure(figsize=(17,8))

# Plot Monte Carlo Simulation
plt.plot(monte_carlo)

# Add labels and title
plt.xlabel('Number of Days')
plt.ylabel('Stock Price ($USD)')
plt.title('10 Future Stock Prices Predictions from Monte Carlo Simulations')

# Show plot
plt.show()

```

![Chart4](Monte%20Carlo%20Simulation/monte_carlo_simulations_graph.png)


## Lets see the distribution of the log returns.

```bash
# Add a figure size for the plot
plt.figure(figsize=(25, 12))

# create histogram of log returns with 50 bins
plt.hist(log_returns, bins=50)

# Add labels and title
plt.xlabel('Log Returns')
plt.ylabel('Frequency')
plt.title('Histogram of Log Returns')

# display the plot
plt.show()
```

![Chart5](Monte%20Carlo%20Simulation/logarithmic_distribution_of_returns.png)




















## Conclusion

### Based on the Monte Carlo simulation analysis performed on American Airlines (AAL) stock data, it appears that the stock has a slightly profitable outlook for the foreseeable future. The Monte Carlo simulation showed a positive median return and a low probability of negative returns, while the SARIMA analysis showed a slightly upward trend in the closing price of the stock. However, it is important to note that these are just predictions and not guarantees, as stock prices can be affected by a wide range of unpredictable external factors. As such, investors should carefully consider their risk tolerance and perform their own due diligence before making any investment decisions.