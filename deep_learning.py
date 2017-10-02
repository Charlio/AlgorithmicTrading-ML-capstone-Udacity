# deep learning
import math
import numpy as np
import pandas as pd
from pylab import plt
plt.style.use('ggplot')
from pandas_datareader import data as web
%matplotlib inline

import tensorflow as tf

data = pd.DataFrame(web.DataReader('AAPL', data_source='google')['Close'])
data.columns = ['prices']
data['log_rets'] = np.log(data['prices'] / data['prices'].shift(1))
data.dropna(inplace=True)
data['Returns'] = np.where(data['log_rets'] > 0, 1, 0)

lags = 10
cols = []
for lag in range(1, lags+1):
    col = 'lag_%d' % lag
    data[col] = data['log_rets'].shift(lag)
    cols.append(col)
data.dropna(inplace=True)

# DNN Classifier
fc = tf.contrib.layers.real_valued_column('lags', dimension=lags)
mean = data['log_rets'].mean()
std = data['log_rets'].std()
fcb = tf.contrib.layers.bucketized_column(fc, boundaries=[mean-std, mean-std/2, mean, mean+std/2, mean+std])

def get_data():
    fc = {'lags': tf.constant(data[cols].values)}
    la = tf.constant(data['returns'].values, shape=[len(data), 1])
    return fc, la

model = tf.contrib.learn.DNNClassifier(hidden_units=[50], feature_columns=[fcb])
%time model.fit(input_fn=get_data, steps=100)
model.evaluate(input_fn=get_data, steps=1)
pred = np.array(list(model.predict(input_fn=get_data)))
data['position'] = np.where(pred > 0, 1, -1)
data['strategy'] = data['position'] * data['log_rets']
# data['position'].value_counts()
# data[['log_rets', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6));
























