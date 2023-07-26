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
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

crypto = 'BTC'
against_crypto = 'USD'
start = datetime.date(2018, 1, 1)
end = datetime.date(2023,7,23)
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
#model(p: number of autoregressive terms(AR order), d:number of nonseasonal differences(differencing order), q:number of moving-average terms(MA order))
ARMAmodel=SARIMAX(train, order=(1, 0, 1))
model_fit = ARMAmodel.fit()

ARMApredictions = model_fit.predict(start=pred_start_date, end=pred_end_date)

#residuals = test-predictions
#print("**********RESIDUALS**********",residuals)
#plt.plot(residuals)

#RMSE(Root Mean Square Error) of testing data compared to predictions, higher number = worse
arma_rmse = numpy.sqrt(mean_squared_error(test, ARMApredictions))
print("ARMA RMSE: ",arma_rmse)
plt.plot(ARMApredictions, color="green", label="ARMA Predictions")

#=========================ARIMA=========================
ARIMAmodel = SARIMAX(train, order=(2,2,2))
ARIMAmodel_fit = ARIMAmodel.fit()
ARIMApredictions = ARIMAmodel_fit.predict(start=pred_start_date, end=pred_end_date)

arima_rmse = numpy.sqrt(mean_squared_error(test, ARIMApredictions))
print("ARIMA RMSE: ",arima_rmse)

plt.plot(ARIMApredictions, color="orange", label="ARIMA Predictions")

#=========================SARIMA=========================
SARIMAmodel = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
SARIMAmodel_fit = SARIMAmodel.fit()
SARIMApredictions = SARIMAmodel_fit.predict(start=pred_start_date, end=pred_end_date)

plt.plot(SARIMApredictions, color="purple", label="SARIMA Predictions")


'''
pred = model_fit.get_forecast(len(test.index))
predictions = pred.conf_int(alpha = 0.05) 
predictions["p"] = model_fit.predict(start=pred_start_date, end=pred_end_date)

predout = predictions["p"] 
print("==========", predout)

plt.plot(predout, color="green", label="Predictions")
'''
#plt.axhline(20000, color="r", linestyle="--", alpha=0.5)

#Graph Labels
plt.legend()
plt.ylabel('BTC Price')
plt.xlabel('Date')

#Styling
plt.xticks(rotation=45)
plt.plot(train, color='black', label="Training")
plt.plot(test, color='red', label="Testing")

plt.show()
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
