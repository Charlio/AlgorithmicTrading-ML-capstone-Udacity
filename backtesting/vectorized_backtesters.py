import numpy as np
import pandas as pd
from pandas_datareader import data as web

symbol = 'AAPL'
data = pd.DataFrame(web.DataReader(symbol, data_source='google'['Close']))
# data.info()

# Strategy Formulation

#Simple Moving Average
data['SMA1'] = data['Close'].rolling(42).mean() # why 42 days?
data['SMA2'] = data['Close'].rolling(252).mean() # one trading year
# data.head()
# data.tail()

''' 
Plot:
from pylab import plt
plt.style.use('ggplot')
%matplotlib inline
data.plot(figsize=(10, 6));
'''

data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)

# data.dropna().plot(figsize=(10, 6), secondary_y='Position')


# Strategy Backtesting
data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
data.dropna(inplace=True)
data['Strategy'] = data['Position'].shift(1) * data['Returns']
data.dropna(inplace=True)
data['CReturns'] = data['returns'].cumsum()
data['CStrategy'] = data['Strategy'].cumsum()

#final_returns = np.exp(data[['CReturns', 'CStrategy']].ix[-1]) - 1
# data[['CReturns', 'CStrategy', 'Position']].apply(np.exp).plot(secondary_y='Position', figsize=(10, 6));


# Transaction Costs
trades = (data['Position'].diff().fillna(0) != 0)
data[trades] # days to trade
tc = 0.01
data['Strategy'] = np.where(trades, data['Strategy'] - tc, data['Strategy'])
data['CStrategy'] = data['Strategy'].cumsum()
final_returns_tc = np.exp(data[['CReturns', 'CStrategy']].ix[-1]) - 1
# data[['CReturns', 'CStrategy', 'Position']].apply(np.exp).plot(secondary_y='Position', figsize=(10, 6));


# Object-oriented design






