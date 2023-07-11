import sys
import matplotlib
import numpy
from numpy import random
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
numpy.random.seed(2)
import datetime
import pandas as pd
import pandas_datareader as web 

from scipy import stats

import yfinance as yf 
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
#from statsmodels.tsa.arima_model import ARMA
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


crypto = 'BTC'
against_crypto = 'USD'
start = datetime.date(2018, 1, 1)
end = datetime.date.today()

btc = yf.download(f'{crypto}-{against_crypto}', start=start, end=end)


btc = btc['Close']

btc.to_csv("btc.csv")

btc = pd.read_csv("btc.csv")


btc.index = pd.to_datetime(btc['Date'], format='%Y-%m-%d')
del btc['Date']

train = btc[btc.index < pd.to_datetime("2022-11-01", format='%Y-%m-%d')]
test = btc[btc.index > pd.to_datetime("2022-11-01", format='%Y-%m-%d')]

#model
model=SARIMAX(train, order=(1, 0, 1))
#fit model
model_fit = model.fit()
print("*******MODEL FIT*******", model_fit.summary())

print(btc.head())
sns.set()
plt.ylabel('BTC Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.plot(train, color='black')
plt.plot(test, color='red')
#plt.plot(btc.index, btc)
plt.show()

'''
x = [1,2,3,4,5,6,7,8,9,10]
y = [10,15,20,25,30,25,30,35,40,50]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

myline = numpy.linspace(1, 10, 50)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


#training and testing data, train 80%, test 20%
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
'''


'''
slope, intercept, r, p, std_err = stats.linregress(x, y)

#0 = no relationship, 1/-1 = full relationship
print("Relationship: ", r)

def linearregression(x):
    return slope * x + intercept
mymodel = list(map(linearregression, x))
'''


'''
#x, y axis labels and x label rotation
plt.ylabel('Y')
plt.xlabel('X')
plt.xticks(rotation=45)

plt.scatter(train_x, train_y)
plt.plot(x, mymodel)
plt.show()


'''
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()