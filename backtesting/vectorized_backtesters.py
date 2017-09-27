import numpy as np
import pandas as pd
from pandas_datareader import data as web


# Simple Moving Average
class SMAVectorizedBacktester(object):
    def __init__(self, symbol, start, end, SMA1, SMA2, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.tc = tc
        self.get_data()
    
    def get_data(self):
        raw = pd.DataFrame(web.DataReader(self.symbol, start=self.start,
                                          end=self.end, data_source='google')['Close'])
        raw['SMA1'] = raw['Close'].rolling(self.SMA1).mean()
        raw['SMA2'] = raw['Close'].rolling(self.SMA2).mean()
        raw['Returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
        self.data = raw.dropna()
        
    def plot_data(self):
        self.data['Close'].plot(figsize(10, 6), title=self.symbol)
        
    def run_strategy(self):
        data = self.data.copy()
        data['Position'] = np.where(data['SMA1'] > data['SMA2'], 1, -1)
        data['Strategy'] = data['Position'].shift(1) * data['Returns']
        data.dropna(inplace=True)
        trades = (data['Position'].diff().fillna(0) != 0)
        data['Strategy'] = np.where(trades, data['Strategy'] - self.tc, data['Strategy'])
        data['CReturns'] = data['Returns'].cumsum().apply(np.exp)
        data['CStrategy'] = data['Strategy'].cumsum().apply(np.exp)
        self.results = data
        return data[['CReturns', 'CStrategy']].iloc[-1] - 1
        
    def plot_results(self):
        self.results[['CReturns', 'CStrategy', 'Position']].plot(secondary_y='Position', figsize=(10, 6))
        
'''        
sma = SMAVectorizedBacktester(symbol='AAPL', start='2012-1-1', end='2017-8-31,
                              SMA1=42, SMA2=252, tc=0.01) # alternative: SMA1=30, SMA2=150
sma.run_strategy()
sma.plot_results()
'''        
        
        
        
    
    




