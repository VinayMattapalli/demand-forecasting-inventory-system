import joblib
from prophet import Prophet
from src.data.load_data import load_data

df = load_data()

temp = df[['Date','Weekly_Sales']]
temp.columns = ['ds','y']

model = Prophet()
model.fit(temp)

joblib.dump(model, "C:/Users/12177/VinayMattapalli/Demand forecasting inventory/models/prophet_model.pkl")

print("Prophet model saved")