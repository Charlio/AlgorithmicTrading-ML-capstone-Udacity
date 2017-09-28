import numpy as np
import pandas as pd
from pandas_datareader import data as web


# Simple Moving Average
class SMAVectorizedBacktester(object):
    def __init__(self, symbol, start, end, tc):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.get_data()
    
    def get_data(self):
        raw = pd.DataFrame(web.DataReader(self.symbol, start=self.start,
                                          end=self.end, data_source='google')['Close'])
        raw['Returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
        self.data = raw.dropna()
        
    def plot_data(self, cols='Close'):
        self.data[cols].plot(figsize(10, 6), title=self.symbol)
        
    def prepare_data(self):
        self.data['SMA1'] = self.data['Close'].rolling(self.SMA1).mean()
        self.data['SMA2'] = self.data['Close'].rolling(self.SMA2).mean()
        
    def run_strategy(self, SMA1, SMA2):
        self.SMA1 = SMA1
        self.SMA2 = SMA2
        self.prepare_data()
        data = self.data.dropna().copy()
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
sma = SMAVectorizedBacktester(symbol='AAPL', start='2012-1-1', end='2017-8-31, tc=0.01) 
sma.run_strategy(SMA1=42, SMA2=252) # alternative: SMA1=30, SMA2=150
sma.plot_results()
'''        

# Brute Force Optimization
import itertools as it
    
                             
# results['CStrategy'] > results['CReturns']
# results['CStrategy'].max()
# iloc = results['CStrategy'].argmax()
# results.iloc[iloc]

class SMAVectorizedOptimizer(SMAVectorizedBacktester):
    def optimize_parameters(self, SMA1_range, SMA2_range):
        self.brute_force = pd.DataFrame()
        for SMA1, SMA2 in it.product(SMA1_range, SMA2_range):
            result = self.run_strategy(SMA1, SMA2)
            self.brute_force = self.brute_force.append(
                                    pd.DataFrame({'SMA1': SMA1, 'SMA2': SMA2,
                                                  'CReturns': result['CReturns'],
                                                  'CStrategy': result['CStrategy']},
                                                 index=[0]),
                                    ignore_index=True)
        return self.bruself.iloc[self.brute_force['CStrategy'].argmax()]
        
# smabf = SMAVectorizedOptimizer(symobl='AAPL', start='2010-1-1', end='2017-8-31', tc=0.01)
# smabf.run_strategy(42, 252)
# smabf.optimize_parameters(range(30, 51, 10), range(220, 261, 10))
# smabf.brute_force

# in-sample and out-of-sample 

smaio = SMAVectorizedOptimizer(symbol='AAPL', start='2010-1-1', end='2014-1-1', tc=0.01)
smaio.optimize_parameters(range(20, 51, 2), range(210, 271, 10))

# 220 days cut out due to SMA2
smaio = SMAVectorizedOptimizer(symbol='AAPL', start='2013-3-31', end='2017-8-31', tc=0.01)
smaio.run_strategy(40, 220)

    




