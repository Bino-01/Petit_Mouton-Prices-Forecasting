import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt

data = pd.read_csv('Petit Mouton.csv')
(printdata.head(7))

dateparse = lambda dates:pd.datetime.strptime(dates, '%d/%m/%Y')
data = pd.read_csv('Petit Mouton.csv',parse_dates=['Date'],index_col='Date',date_parser=dateparse)
print('\n Parsed Data:')
print(data.head(7))

from datetime import datetime

data.index
#defining the time-series=ts as function and then printing the first 7 rows from the head

data=pd.read_csv('Petit Mouton.csv', index_col=0, parse_dates=True)

data['Year'] = data.index.year
data['Month'] = data.index.month
data['Day'] = data.index.day
data.sample(7, random_state=0)

data.loc['2010-01-31']
data.loc['2009-01-31': '2013-10-30']

ts = data['Price Vintage 2007'] 
ts.head(7)
ts['2010-01-31']

plt.plot(ts)
plt.xlabel('year', fontsize = 15)
plt.ylabel('price', fontsize = 15)

# The Petit Mouton vintage 2009 plot shows an .... pattern of rises and falls but which takes nearly 2 years. 
# Therefore, it is a cyclic Time-Series, with a decending trend.
# Let's  a Dickey-Fuller Test to find out more about this Time Series whether it is stationary or not. 
# If Test Statistic ? Critic Values, then null hypothesis will be rejected and I will conclude that the Seires is stationary. 

from statsmodels.tsa.stattools import adfuller
def test_stationarity(ts):

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import statsmodels.api as sm
    
rolmean = pd.Series(ts).rolling(window=12).mean()

rolstd = pd.Series(ts).rolling(window=12).std()

#Plot rolling statistics:
orig = plt.plot(ts, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation - Petit Mouton 2007')
plt.show(block=False)
	
#Perform Dickey-Fuller test:
# print ('Results of Dickey-Fuller Test:'
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

# Test Statistic > Critical Values. Therefore the null hypothesis is not rejected and the Petit Mouton 2007 TS is Stationary.

test_stationarity(ts)

# The rolling standard deviation is varying between 800. 
# On the other hand, the rolling mean varies a lot between 2400 & 4400, with a descending Trend. 
# It is also displaying rises and falls pattern. 

# Eliminate Trend and stationarity
ts_log = np.log(ts)
plt.plot(ts_log)
plt.title('Plot Log TS - Petit Mouton 2007', fontsize = 15)
plt.xlabel('year', fontsize = 15)
plt.ylabel('ts_log', fontsize = 15)

# There is an increasing Trend to be noticed and a cyclic pattern.

# Moving Average = Taking the past few instances and
# removing the trend that can be seen here as noise.

moving_avg = ts_log.rolling(12).mean()
plt.plot(ts_log)
plt.plot(moving_avg, color='red')
plt.title('Plot Log TS & Moving Average - Petit Mouton 2007', Fontsize = 15)
plt.xlabel('year', fontsize = 15)
plt.ylabel('dprice', fontsize = 15)
# The red line is the plot of the moving average

ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.head(15)
# As we are taking the average of the last 24 values due to the fact that we are closer to 2 years = 24 months
# the rolling mean will be defined in our case for the first 11 values, but we are viewing the 15 first values. 

ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)
# print(test_stationarity(ts_log_moving_avg_diff)


# We notice a large variation of the Rolling Std
# The rolling mean also dislays up and down Trend with an overall ascendant Trend. 
# Let's run another Dickey Fuller Test and assess the stationarity


dftest = adfuller(ts_log_moving_avg_diff, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

# Dickey-Fuller Test Statistic < Critical Value (5%) Therefore we can state that the TS is now Stationary with 95% confidence.
# This has significantly improved the Time Series Stationarity

	
expwighted_avg = pd.Series(ts_log)
expwighted_avg.ewm(halflife=12).mean()
plt.plot(ts_log, color = 'blue')
plt.plot(expwighted_avg, linestyle='--', color = 'yellow')
plt.title ('Plot Log Time Series & Exponential Weighted -Petit Mouton 2007', fontsize = 15)
plt.xlabel('year', fontsize = 15)
plt.ylabel('price', fontsize = 15)	


# We clearly notice that both log_ts and the expwighted_avg have the same plot/identical.
# The color obtained is a mix of both color blue and green - Both plots are the same
	
# Eliminating the seasonality

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.title ('Plot Log Diff TS -Petit Mouton 2007', fontsize = 15)

# We notice tat this has reduced quite significantly the Trend, which can be verified using 

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# We notice now less variation of the std and the rolling mean espeially the latter variation has significantly decrease 

dftest = adfuller(ts_log_diff, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print (dfoutput)

# The new Dickey-Fuller Test Statistics  shows that the value of this test has decreased from (-3.325916) to (-3.338315)
# Now let's decompose the whole function into the different components, including the Trend part, the Seasonality part and 
# finally the residual part 


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.title ('Original Petit Mouton 2007', fontsize = 15)
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.title ('Trend Petit Mouton 2007', fontsize = 15)
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.title ('Seasonaity Petit Mouton', fontsize = 15)
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.title ('Residual Petit Mouton 2007', fontsize = 15)
plt.tight_layout()

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf

ts_log_diff = np.log(ts)
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

# We can determine p and q values from the Autocorrelation Function (ACF) and the Partial Autocorrelation Function (PACF): p = 2 and q = 2

# Loading the ARIMA model
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %0.4f'% sum((results_AR.fittedvalues-ts_log_diff)**2))

model = ARIMA(ts_log, order=(2, 1, 0))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))

predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (predictions_ARIMA_diff.head(7))

predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head(7)

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts, color = '#4254f5')
plt.plot(predictions_ARIMA, color = '#75f542')
plt.title('Petit Mouton - RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
