import pandas as pd
from pandas_datareader import data as web
stock_names = ['AAPL', 'GOOG', 'MSFT', 'FB', 'AMZN', 'NVDA']


h5 = pd.HDFStore('data.h5', 'w')

for stock_name in stock_names:
    data = web.DataReader(name=stock_name, data_source='yahoo')['Adj Close']
    data = pd.DataFrame(data)
    data.rename(columns={'Adj Close': 'price'}, inplace=True)
    print(data.info())
    h5[stock_name] = data
h5.close()
    
    