import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("Data_Science/Projects/Reliance.csv")
print(df.head(5))
print(df.info())
print(df.isnull().sum())
#==== as their are only 1 null values in each feature =======
df.dropna(inplace=True)
print(df.isnull().sum())

#==== Converting the date column into the datetime format ======
df["Date"] = pd.to_datetime(df["Date"])
print(df.head(5))

#===== visualizations ======
plt.figure(figsize=(10,8))
plt.plot(df["Date"], df["Open"], label="Opening Price",color='Red')
plt.plot(df["Date"], df["Adj Close"], label="Closing Price",color='black')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Opening vs Closing Stock Price")
plt.legend()
plt.show()

#===== Checking for the seasonality =====
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print("Results of the Dickey Fuller Test")
    dftest =adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput= pd.Series(
        dftest[0:4],
        index=[
            'Test Statistic',
            'p-value',
            '#Lag used',
            'Number of observations used'
        ]
    )
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value
    print(dfoutput)
print('ADF for Adj Close')
adf_test(df["Adj Close"])
result = adfuller(df["Adj Close"].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("Data is Stationary")
else:
    print("Data is Non-Stationary")

#===== As the p-value > 0.05 hence we need to apply transformations =======
#===== Again applying the ADF test on the transformed part ========

df['Adj_close_diff'] = df["Adj Close"].diff()

print('ADF for Adj_close_diff')
adf_test(df["Adj_close_diff"])
result =adfuller(df["Adj_close_diff"].dropna())
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("Data is Stationary")
else:
    print("Data is Non-Stationary")

#===== As now the data is stationary, it is relevant for the ARIMA model ======
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
data = df["Adj_close_diff"].dropna()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
plot_acf(data, lags=50, ax=ax1)
plot_pacf(data, lags=50, ax=ax2)
plt.tight_layout()
plt.show()

#===== Using the Moving average model as the baseline model =======
df["MA_20"]=df["Adj Close"].rolling(window=20).mean()
df["MA_50"]=df["Adj Close"].rolling(window=50).mean()
plt.figure(figsize=(12,6))
plt.plot(df["Date"],df["Adj Close"], label="Actual Price")
plt.plot(df["Date"],df["MA_20"],label="20-Day MA")
plt.plot(df["Date"],df["MA_50"],label="50-Day MA")
plt.title("Moving Average Analysis")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

#===== chossing the values of p and q with order=(1,1,1)
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

#=== with order(1,1,1)=====
train_size=int(len(df["Adj Close"]) * 0.8)
train_data=df["Adj Close"][:train_size]
test_data=df["Adj Close"][train_size:]
model=ARIMA(train_data, order=(1,1,1))
model_fit=model.fit()
pred =model_fit.forecast(steps=len(test_data))
rmse=sqrt(mean_squared_error(test_data,pred))
mae =mean_absolute_error(test_data, pred)
print("RMSE:",rmse)
print("MAE:",mae)
plt.figure(figsize=(12,8))
plt.plot(df["Date"][train_size:] ,test_data,label='actual')
plt.plot(df["Date"][train_size:],pred,label='predicted')
plt.xlabel('Date')
plt.ylabel('Price_values')
plt.title('Acutal v/s predicted')
plt.legend()
plt.show()

# ===== Future Forecasting =====
future_steps = 30
future_forecast=model_fit.forecast(steps=future_steps)
future_dates=pd.date_range(
    start=df["Date"].iloc[-1],
    periods=future_steps+1,
    freq='B'
)[1:]
future_df =pd.DataFrame({
    "Date":future_dates,
    "Predicted Price":future_forecast
})
print(future_df)

#===== Visualizing =======
plt.figure(figsize=(12,6))
plt.plot(df["Date"], df["Adj Close"], label="Historical Prices")
plt.plot(
    future_df["Date"],
    future_df["Predicted Price"],
    label="Future Forecast",
    color='red'
)
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Future Stock Price Forecast")
plt.legend()
plt.show()

#===== using the random forest method (ML model) for the predictive part =========
# Random Forest Regressor was used to capture nonlinear relationships and complex patterns in stock price
# movement that traditional statistical models like ARIMA could not model effectively.
df["Lag_1"]=df["Adj Close"].shift(1)
df["Lag_2"]=df["Adj Close"].shift(2)
df["Lag_3"]=df["Adj Close"].shift(3)
df.dropna(inplace=True)
# ===== Features and Target =====
X = df[["Lag_1","Lag_2","Lag_3"]]
y = df["Adj Close"]
train_size = int(len(df) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print("RMSE:", rmse)
print("MAE : ",mae)
# “Why did you create lag features?”
# Random Forest is not inherently a time-series model, so historical observations were transformed into lag-based supervised learning features.

#====== Now using the XG BOOST (ML boosting model) ======
from xgboost import XGBRegressor

xgb_model= XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb_model.fit(X_train,y_train)
xgb_pred=xgb_model.predict(X_test)
xgb_rmse=sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae=mean_absolute_error(y_test, xgb_pred)
print("XGBoost RMSE:",xgb_rmse)
print("XGBoost MAE:",xgb_mae)

plt.figure(figsize=(12,8))
plt.plot(df["Date"][train_size:], y_test, label='Actual')
plt.plot(df["Date"][train_size:], xgb_pred, label='XGBoost Predicted')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('XGBoost: Actual vs Predicted')
plt.legend()
plt.show()













