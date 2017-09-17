import pandas as pd
from pandas_datareader import data as web

stock_names = ['AAPL', 'GOOG', 'MSFT', 'FB', 'AMZN', 'NVDA']

h5 = pd.HDFStore('data.h5', 'r')

for stock_name in stock_names:
    data = pd.DataFrame(h5[stock_name])
    print(data.info())
h5.close()
