import sys
import matplotlib
import numpy
from numpy import random
from sklearn.metrics import mean_squared_error
numpy.random.seed(2)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
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
#end = datetime.date.today()

btc = yf.download(f'{crypto}-{against_crypto}', start=start, end=end)
btc = btc['Close']
btc.to_csv("btc.csv")
btc = pd.read_csv("btc.csv")
btc.index = pd.to_datetime(btc['Date'], format='%Y-%m-%d')
del btc['Date']

train = btc[btc.index < pd.to_datetime("2022-11-01", format='%Y-%m-%d')]
test = btc[btc.index > pd.to_datetime("2022-11-01", format='%Y-%m-%d') ]
plt.figure(figsize=(10,4))

#Prediction data
pred_start_date = test.index[0]
pred_end_date = test.index[-1]

#=========================ARMA=========================
#model, 
#(p: number of autoregressive terms(AR order), d:number of nonseasonal differences(differencing order), q:number of moving-average terms(MA order))
ARMAmodel=SARIMAX(train, order=(1, 0, 1))
model_fit = ARMAmodel.fit()
#print("*******MODEL FIT*******", model_fit.summary())
#print("*******Head*******", btc.head())

ARMApredictions = model_fit.predict(start=pred_start_date, end=pred_end_date)


#residuals = test-predictions
#print("*************",residuals)
#print("*************",predictions)
#plt.plot(residuals)

#Calculating Root Mean Square Error(RMSE) of testing data compared to predictions, higher number = worse
arma_rmse = numpy.sqrt(mean_squared_error(test, ARMApredictions))
print("ARMA RMSE: ",arma_rmse)
plt.plot(ARMApredictions, color="green", label="ARMA Predictions")
#=========================ARIMA=========================
ARIMAmodel = SARIMAX(train, order=(2,2,2))
ARIMAmodel_fit = ARIMAmodel.fit()
ARIMApredictions = ARIMAmodel_fit.predict(start=pred_start_date, end=pred_end_date)

#arima_rmse = numpy.sqrt(mean_squared_error(test, ARIMApredictions))
#print("ARIMA RMSE: ",arima_rmse)

plt.plot(ARIMApredictions, color="orange", label="ARIMA Predictions")

'''
#=========================SARIMA=========================
SARIMAmodel = SARIMAX(train, order=(5,4,2), seasonal_order=(2,2,2,12))
SARIMAmodel_fit = SARIMAmodel.fit()
SARIMApredictions = SARIMAmodel_fit.predict(start=pred_start_date, end=pred_end_date)

plt.plot(SARIMApredictions, color="red", label="SARIMA Predictions")
'''

'''
pred = model_fit.get_forecast(len(test.index))
predictions = pred.conf_int(alpha = 0.05) 
predictions["p"] = model_fit.predict(start=pred_start_date, end=pred_end_date)
#residuals = test-predictions
#print("============",residuals)


predout = predictions["p"] 
print("==========", predout)

#plt.plot(residuals)
plt.plot(predout, color="green", label="Predictions")
'''
#plt.axhline(20000, color="r", linestyle="--", alpha=0.5)

plt.legend()


#sns.set()
#graph labels
plt.ylabel('BTC Price')
plt.xlabel('Date')
#styling
plt.xticks(rotation=45)
plt.plot(train, color='black', label="Training")
plt.plot(test, color='red', label="Testing")
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