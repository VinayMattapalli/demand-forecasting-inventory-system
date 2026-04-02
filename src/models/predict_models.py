import joblib
from src.data.load_data import load_data
from src.features.build_features import build_features
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

def forecast_next_30_days(prophet_model):

    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)

    return forecast[['ds', 'yhat']].tail(30)

prophet_model = joblib.load("models/prophet_model.pkl")
future_forecast = forecast_next_30_days(prophet_model)
def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse
def predict_all():

    df = load_data()
    df = build_features(df)

    # XGBoost
    xgb = joblib.load("C:/Users/12177/VinayMattapalli/Demand forecasting inventory/models/xgb_model.pkl")
    sample = df.iloc[-1:]
    X = sample[['year','month','week']]
    xgb_pred = xgb.predict(X)[0]

    # ARIMA
    arima = joblib.load("C:/Users/12177/VinayMattapalli/Demand forecasting inventory/models/arima_model.pkl")
    arima_pred = arima.forecast(steps=1).iloc[0]

    # Prophet
    prophet = joblib.load("C:/Users/12177/VinayMattapalli/Demand forecasting inventory/models/prophet_model.pkl")
    future = prophet.make_future_dataframe(periods=1)
    forecast = prophet.predict(future)
    prophet_pred = forecast['yhat'].iloc[-1]

    return xgb_pred, arima_pred, prophet_pred
