# Event-based backtesting demo
import math
import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('ggplot')
from pandas_datareader import data as web
%matplotlib inline

data = web.DataReader('AAPL', data_source='google')


# Simple Base Class
class BacktestBase(object):
    def __init__(self, symbol, start, end, amount, ftc=0.0, ptc=0.0):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.initial_amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.trades = 0
        self.units = 0
        self.position = 0 #neural position
        self.get_data()
        
    def get_data(self):
        raw = pd.DataFrame(web.DataReader(self.symbol, start=self.start,
                                          end=self.end, data_source='google')['Close'])
        raw['Returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
        self.data = raw.dropna()
        
    def plot_data(self, cols='Close'):
        self.data[cols].plot(figsize(10, 6), title=self.symbol)
        
    def print_balance(self, date=''):
        print('%s | current cash balance %9.2f' % (date, self.amount))
        
    def get_date_price(self, bar):
        date = str(self.data.index[bar])[:10]
        price = self.data['Close'].iloc[bar]
        return date, price
    
    def place_buy_order(self, bar, units=None, amount=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = math.floor(amount / price) # ftc, ptc ? 
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        print('%s | buying %4d units at %8.2f' % (date, units, price))
        self.print_balance(data)
        
    def place_sell_order(self, bar, units=None, amount=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = math.floor(amount / price) # ftc, ptc ? 
        self.amount -= (units * price) * (1 + self.ptc) + self.ftc
        self.units += units
        self.trades += 1
        print('%s | buying %4d units at %8.2f' % (date, units, price))
        self.print_balance(data)
        
        
        
        
        
    
# bb = BacktestBase('AAPL', '2010-1-1', '2017-8-31', 1000)
# bb.plot_data()
# bb.print_balance()
# bb.get_date_price(10)
# bb.place_buy_order(bar=10, units=100)
    

    
    
    
    
    
    
    
    
