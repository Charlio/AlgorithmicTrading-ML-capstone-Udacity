# Linear Regression Prediction

import math
import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('ggplot')
from pandas_datareader import data as web
%matplotlib inline

data = web.DataReader('SPY', data_source='google')
reg = np.polyfit(np.arange(len(data)), data['Close'].values, deg=1)
ols = np.polyval(reg, np.arange(len(data)))
data['OLS'] = ols
# data[['Close', 'OLS']].plot(figsize=(10, 6));
data['Position'] = np.where(data['Close'] < data['OLS'], 1, -1) # future data is used, not reliable: foresight, only a sandbox example
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data['Strategy'] = data['Position'].shift(1) * data['Returns']
# data[['Returns', 'Strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))

# working with intervals
days = 100
ols = []
for i in range(days, len(data), days):
    reg = np.polyfit(np.arange(days), data['Close'].values[i-days:i], deg=1)
    values = list(np.polyval(reg, np.arange(days)))
    ols.extend(values)
data['OLS'] = np.nan
data.iloc[:len(ols), 'OLS'] = ols    




# price prediction with linear regression

data = web.DataReader('SPY', data_source='google')
data.dropna(inplace=True)
lags = 3 # random walk hypothesis: the last one primary
cols = []
for lag in range(1, lags+1):
    col = 'Lab_%s' % lag
    data[col] = data['Close'].shift(lag)
    cols.append(col)
colsc = ['Close']
colsc.extend(cols)  
temp = data[cols].dropna().iloc[:5]
# temp = data[colsc].dropna().iloc[:5]
# temp['Close'].values
# temp[cols].values

reg = np.linalg.lstsq(temp[cols], temp['Close'])
np.dot(temp[cols], reg).round(2)

# full dataset
data.dropna(inplace=True)
reg = np.linalg.lstsq(data[cols].values, data['Close'].values)[0]
ols = np.dot(data[cols].values, reg)
data['OLS'] = ols
data[['Close', 'OLS']].iloc[-50:].plot(figsize=(10, 6));

# TODO
# implement out-of-sample testing for the regression model, see sma


# predicting market direction
data = web.DataReader('SPY', data_source='google')
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
lags = 5
cols = []
for lag in range(1, lags+1):
    col = 'Lag_%s' % lag
    data[col] = data['Returns'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)
reg = np.linalg.lstsq(data[cols].values, data['Returns'].values)[0] # using log returns directly
# reg = np.linalg.lstsq(data[cols].values, np.sign(data['Returns'].values))[0] # using the sign of the log returns instead
ols = np.dot(data[cols], reg)
ols.round(4)
data['OLS'] = ols

# data[['Returns', 'OLS']].iloc[50:].plot(figsize=(10, 6));
# data[['Returns', 'OLS']].mean() * 252
# data['Returns'].std() * 252 ** 0.5 # annualized volatility
# data['OLS'].std() * 252 ** 0.5 # much lower volatility

# hit ratio 
res = np.sign(data['Returns'] * data['OLS']).value_counts() 
# res # about 50% correct predictions of directions
data['Position'] = np.sign(data['OLS'])

# more important to hit the important ones correctly
data['Strategy'] = data['Position'] * data['Returns']
data[['Returns', 'Strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6));








    
    
    
    
    
    
    
    










