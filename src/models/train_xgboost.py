import joblib
from xgboost import XGBRegressor
from src.data.load_data import load_data
from src.features.build_features import build_features

df = load_data()
df = build_features(df)

features = ['year','month','week']
target = 'Weekly_Sales'

X = df[features]
y = df[target]

model = XGBRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "C:/Users/12177/VinayMattapalli/Demand forecasting inventory/models/xgb_model.pkl")

print("XGBoost model saved")