# machine learning approach

import math
import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('ggplot')
from pandas_datareader import data as web
%matplotlib inline

from sklearn  import linear_model
from sklearn import preprocessing

data = web.DataReader('SPY', data_source='google')['Close']
data = pd.DataFrame(data)
data['Returns'] = np.log(data / data.shift(1))

lags = 5
cols = []
for lag in range(1, lags + 1):
    col = 'Lag_%d' % lag
    data[col] = data['Returns'].shift(lag)
    cols.append(col)
    
data.dropna(inplace=True)

# Logistic Regression

lm = linear_model.LogisticRegression(C=1e6)
lm.fit(data[cols], np.sign(data['Returns']))
pred = lm.predict(data[cols])
data['Position'] = pred
data['Strategy'] = data['Position'] * data['Returns']
# data['Position'].value_counts()
# data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));

# Buckets
mean = data['Returns'].mean()
std = data['Returns'].std()
def buckets(x):
    v = 0
    for b in [mean - std, mean - std/2, mean, mean + std/2, mean + std]:
        if x < b: return v
        v += 1
    return value_counts
    
# data['Lag_1'].apply(lambda x: buckets(x)).value_counts()
for col in cols:
    data[col] = data[col].apply(lambda x: buckets(x))
    
lm.fit(data[cols], np.sign(data['Returns']))
pred = lm.predict(data[cols])
data['Position'] = pred
data['Strategy'] = data['Position'] * data['Returns']
# data['Position'].value_counts()
# data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));


# more features

data = web.DataReader('SPY', data_source='google')['Close']
data = pd.DataFrame(data)
data['Returns'] = np.log(data / data.shift(1))

lags = 15
cols = []
for lag in range(1, lags + 1):
    col = 'Lag_%d' % lag
    data[col] = data['Returns'].shift(lag)
    cols.append(col)
    
# simple moving average    
data['SMA1'] = data['Close'].rolling(10).mean()
data['SMA2'] = data['Close'].rolling(60).mean()
data['SMA'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
data['SMA'] = data['SMA'].shift(1)
cols.append('SMA')
# data['SMA'].value_counts()

# momentum
data['MOM'] = np.where(data['Returns'].rolling(10).mean() > 0, 1, 0)
data['MOM'] = data['MOM'].shift(1)
cols.append('MOM')
# data['MOM'].value_counts()

data.dropna(inplace=True)

# logistic regression
lm.fit(data[cols], np.sign(data['Returns']))
pred = lm.predict(data[cols])
data['Position'] = pred
data['Strategy'] = data['Position'] * data['Returns']
# data['Position'].value_counts()
# data[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));

# in-sample and out-of-sample
data = web.DataReader('GLD', data_source='google')['Close']
data = pd.DataFrame(data)
data['Returns'] = np.log(data / data.shift(1))

lags = 20
cols = []
for lag in range(1, lags + 1):
    col = 'Lag_%d' % lag
    data[col] = data['Returns'].shift(lag)
    cols.append(col)
    
data['SMA1'] = data['Close'].rolling(20).mean()
data['SMA2'] = data['Close'].rolling(60).mean()
data['SMA'] = np.where(data['SMA1'] > data['SMA2'], 1, 0)
data['SMA'] = data['SMA'].shift(1)
cols.append('SMA')
# data['SMA'].value_counts()

for m in [5, 10, 15]:
    col = 'MOM_%d' % ma
    data[col] = np.where(data['Returns'].rolling(m).mean() > 0, 1, 0)
    data[col] = data[col].shift(1)
    cols.append(col)
data.dropna(inplace=True)

# model traning
cutoff = '2015-1-1'
train = data[data.index < cutoff].copy()
lm.fit(train[cols], np.sign(train['Returns']))
pred = lm.predict(train[cols])
train['Position'] = pred
train['Strategy'] = train['Position'] * train['Returns']
# train['Position'].value_counts()
# train[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));

# model testing
cutoff = '2015-1-1'
test = data[data.index >= cutoff].copy()
pred = lm.predict(test[cols])
test['Position'] = pred
test['Strategy'] = test['Position'] * test['Returns']
# test['Position'].value_counts()
# test[['Returns', 'Strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));
































































