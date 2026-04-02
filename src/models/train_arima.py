import joblib
from statsmodels.tsa.arima.model import ARIMA
from src.data.load_data import load_data

df = load_data()

model = ARIMA(df['Weekly_Sales'], order=(5,1,0))
model_fit = model.fit()

joblib.dump(model_fit, "C:/Users/12177/VinayMattapalli/Demand forecasting inventory/models/arima_model.pkl")

print("ARIMA model saved")