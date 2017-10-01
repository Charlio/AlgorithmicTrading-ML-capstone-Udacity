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











