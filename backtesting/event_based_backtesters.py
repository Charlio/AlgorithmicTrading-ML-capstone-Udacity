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
    def __init__(self, symbol, start, end, amount, ftc=0.0, ptc=0.0, verbose=True):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.amount = amount
        self.initial_amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.verbose = verbose
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
        if self.verbose is True:
            print('%s | buying %4d units at %8.2f' % (date, units, price))
        self.print_balance(date)
        
    def place_sell_order(self, bar, units=None, amount=None):
        date, price = self.get_date_price(bar)
        if units is None:
            units = math.floor(amount / price) # ftc, ptc ? 
        self.amount += (units * price) * (1 - self.ptc) - self.ftc
        self.units -= units
        self.trades += 1
        if self.verbose is True:
            print('%s | selling %4d units at %8.2f' % (date, units, price))
        self.print_balance(date)
        
    def close_out(self, bar):
        date, price = self.get_date_price(bar)
        self.amount += (self.units * price) # include ftc/ptc ?
        print(50 * '=')
        print('%s | buying/selling %4d units at %8.2f' % (date, self.units, price))
        print('Final balance [$]: %8.2f' % self.amount)
        perf = (self.amount - self.initial_amount) / self.initial_amount * 100
        print('Performance  [%%]: %8.2f' % perf)
        print('Trades       [#]: %8d' % self.trades)
        
        
    
# bb = BacktestBase('AAPL', '2010-1-1', '2017-8-31', 1000)
# bb.plot_data()
# bb.print_balance()
# bb.get_date_price(10)
# bb.place_buy_order(bar=10, units=100)
# bb.place_sell_order(bar=20, units=50)    
# bb.place_sell_order(bar=20, amount=1000)
# bb.close_out(50)


class BacktestLongOnly(BacktestBase):
    def run_sma_strategy(self, SMA1, SMA2):
        print('Running SMA strategy for %s with SMA1 = %d and SMA2 = %d' 
              $ (self.symbol, SMA1, SMA2))
        print(60 * '=')
        self.position = 0
        self.units = 0
        self.trades = 0
        self.amount = self.initial_amount
        self.data['SMA1'] = self.data['Close'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['Close'].rolling(SMA2).mean()
        
        for bar in range(len(self.data)):
            if bar >= SMA2:
                if self.position == 0:
                    if self.data['SMA1'].iloc[bar] > self.data['SMA2'].iloc[bar]:
                        # self.place_buy_order(bar, units=50) # number of units
                        # self.place_buy_order(bar, amount=5000) # fixed amount
                        self.place_buy_order(bar, amount = 0.8 * self.amount) # variable amount
                        self.position = 1
                elif self.position == 1:
                    if self.data['SMA1'].iloc[bar] < self.data['SMA2'].iloc[bar]:
                        self.place_sell_order(bar, units=self.units)
                        self.position = 0
        self.close_out(bar)
        
# smalo = BacktestLongOnly('AAPL', '2010-1-1', '2017-8-31', 10000, 10, 0.01)
# smalo.run_sma_strategy(42, 252)


# Momentum Strategy

# data = web.DataReader('GLD', data_source='google')
# momentum = 3
# data['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
# data['Momentum'] = np.sign(data['Returns'].rolling(momentum).mean())
# data['Strategy'] = data['Momentum'].shift(1) * data['Returns']
# data[['Returns', 'Strategy']].dropna().cumsum().apply(np.exp).plot(figsize(10, 6)) 

class BacktestLongOnly(BacktestLongOnly):
    def run_mom_strategy(self, momentum):
        print('\nRunning momentum strategy for %s with momentum = %d'
              % (self.symbol, momentum))
        print(60 * '=')
        self.position = 0
        self.units = 0
        self.trade = 0
        self.amount = self.initial_amount
        self.data['Momentum'] = np.sign(self.data['Returns'].rolling(momentum).mean())
        
        for bar in range(len(self.data)):
            if bar >= momentum:
                if self.position == 0:
                    if self.data['Momentum'].iloc[bar] > 0:
                        self.place_buy_order(bar, amount=0.9 * self.amount)
                        self.position = 1
                elif self.position == 1:
                    if self.data['Momentum'].iloc[bar] < 0:
                        self.place_sell_order(bar, units = self.units)
                        self.position = 0
        self.close_out(bar)

# momlo = BacktestLongOnly('GLD', '2010-1-1', '2017-8-31', 10000, 10, 0.01, False)
# for momentum in [3, 5, 7, 9, 11, 13, 15]:
#   momlo.run_mom_strategy(momentum)



# Long short backtesting class

class BacktestLongShort(BacktestBase):
    def run_mom_strategy(self, momentum):
        print('\nRunning momentum strategy for %s with momentum = %d' 
              $ (self.symbol, momentum))
        print(60 * '=')
        self.position = 0
        self.units = 0
        self.trades = 0
        self.amount = self.initial_amount
        self.data['SMA1'] = self.data['Close'].rolling(SMA1).mean()
        self.data['SMA2'] = self.data['Close'].rolling(SMA2).mean()
        
        for bar in range(len(self.data)):
            if bar >= momentum:
            
                if self.position == 0:
                    if self.data['Momentum'].iloc[bar] > 0:
                        sefl.place_buy_order(bar, amount = 0.95 * self.amount) # variable amount
                        self.position = 1
                    elif self.data['Momentum'].iloc[bar] < 0:
                        self.place_sell_order(bar, amount=0.95 * self.amount)
                        self.position = -1
                        
                elif self.position == 1:
                    if self.data['Momentum'].iloc[bar] < 0:
                        self.place_sell_order(bar, units=self.units)
                        self.place_sell_order(bar, amount=0.95 * self.amount)
                        self.position = -1
                        
                elif self.position == -1:
                    if self.data['Momentum'].iloc[bar] > 0:
                        self.place_buy_order(bar, units=-self.units)
                        self.place_buy_order(bar, amount=0.95 * self.amount)
                        self.position = 1
        self.close_out(bar)  
        
# momls = BacktestLongShort('AAPL', '2010-1-1', '2017-8-31', 10000, 0, 0.0, True)
# momls.run_mom_strategy(50)

    
    
    
    
    
    
    
    
