import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv("Data_Science/data/TimeSeries_TotalSolarGen_and_Load_IT_2016.csv")
print(df.info())
print(df.head(5))

#====== converting to datatime =======
# df['utc_timestamp']=pd.to_datetime(df['utc_timestamp'])
# plt.figure(figsize=(12,8))
# plt.plot(df['utc_timestamp'],df['IT_load_new'],label='Load')
# plt.plot(df['utc_timestamp'],df['IT_solar_generation'],label='Solar_genration')
# plt.xlabel('Time')
# plt.ylabel('Load_value')
# plt.legend()
# plt.show()

# #==== Hnadling the missing values =====
# print(df.isnull().sum())
# print("\n")
# df["IT_load_new"].fillna(method='ffill',inplace=True)
# print(df.isnull().sum())

# #===== Using decompostion method to check for seasonality, trends and residuals =======
# from statsmodels.tsa.seasonal import seasonal_decompose
# results=seasonal_decompose(df['IT_load_new'],model='additive',period=168)  #===== weekly tracking= 24*7
# plt.rcParams.update({'figure.figsize': (8,6)})
# results.plot()
# plt.show()

# #===== checking for the stationarity =====
# #=== using the dickey fuller test =======
# from statsmodels.tsa.stattools import adfuller
# def adf_test(timeseries):
#     print("Results of the dickey fuller test ")
#     dftest=adfuller(timeseries,autolag='AIC')
#     dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lag used','Number of observations used'])
#     for key,value in dftest[4].items():
#         dfoutput['Critical value (%s)'%key]=value
#     print(dfoutput)

# print("ADF for the IT_load_new")
# adf_test(df['IT_load_new'])

# print("ADF for the IT_solar_generation")
# adf_test(df['IT_solar_generation'])

# # #=== As the p-value is very much less hence the data is stationary ======
# #===== using the plots ACF adn PACF for the values of p and q =====
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# data = df['IT_load_new'].diff().dropna()
# fig,(ax1,ax2)=plt.subplots(2, 1 ,figsize=(9,6))
# plot_acf(data,lags=50,ax=ax1,zero=False)
# plot_pacf(data,lags=50,ax=ax2,zero=False)
# plt.show()


# #=== hence from the model we get p=2 and q=2====
# # #====== Using the ARIMA model =======
# from statsmodels.tsa.arima.model import ARIMA
# from sklearn.metrics import mean_squared_error
# from math import sqrt

# #=== with order(2,0,2)
# train_data=int(len(df['IT_load_new'])*0.8)
# train,test=df['IT_load_new'][:train_data],df['IT_load_new'][train_data:]
# model=ARIMA(train,order=(2,1,2))
# model_fit=model.fit()
# pred=model_fit.predict(start=len(train),end=len(train)+len(test)-1)
# rmse=sqrt(mean_squared_error(test,pred))
# mae = mean_absolute_error(test, pred)
# print(rmse)

# plt.figure(figsize=(12,8))
# plt.plot(df['utc_timestamp'][train_data:],test,label='actual')
# plt.plot(df['utc_timestamp'][train_data:],pred,label='predicted')
# plt.xlabel('Time')
# plt.ylabel('Load_value')
# plt.title('Acutal v/s predicted')
# plt.legend()
# plt.show()

# #====== Using the SariMax model ========
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# sarima_model = SARIMAX(train,order=(2, 1, 2),seasonal_order=(1, 1, 1, 24))
# sarima_fit = sarima_model.fit(disp=False)
# sarima_pred = sarima_fit.predict(start=len(train),end=len(train) + len(test) - 1)

# sarima_rmse = sqrt(mean_squared_error(test, sarima_pred))
# sarima_mae = mean_absolute_error(test, sarima_pred)
# print("SARIMA Results")
# print("RMSE:",sarima_rmse)
# print("MAE:",sarima_mae)

#===== we should use the prophet =======
from prophet import Prophet
df_prophet = df[['utc_timestamp', 'IT_load_new']].copy()
df_prophet['utc_timestamp'] = pd.to_datetime(df_prophet['utc_timestamp']).dt.tz_localize(None)
df_prophet.rename(columns={
    'utc_timestamp': 'ds',
    'IT_load_new': 'y'
}, inplace=True)
prophet_model = Prophet()
prophet_model.fit(df_prophet)
# Forecast next 24 hours
future = prophet_model.make_future_dataframe(periods=24, freq='H')
forecast = prophet_model.predict(future)
prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

print("""
Project Conclusion:
- ARIMA underfits due to strong seasonality
- SARIMA performs better because it handles seasonality
- Prophet provides strong forecasting with automatic seasonality detection
- Best practical models for this dataset: SARIM""")