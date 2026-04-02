from src.data.load_data import load_data
from src.models.predict_models import predict_all, evaluate_model
from src.ensemble.ensemble import ensemble_forecast
from src.inventory.inventory_model import optimize_inventory

df = load_data()

# Split for evaluation
train = df[:-10]
test = df[-10:]

actual = test['Weekly_Sales']

xgb, arima, prophet = predict_all()

final_forecast = ensemble_forecast(xgb, arima, prophet)

# Fake predictions for evaluation (simple repeat)
xgb_preds = [xgb] * len(actual)
arima_preds = [arima] * len(actual)
prophet_preds = [prophet] * len(actual)
ensemble_preds = [final_forecast] * len(actual)

# Evaluate
xgb_mae, xgb_rmse = evaluate_model(actual, xgb_preds)
arima_mae, arima_rmse = evaluate_model(actual, arima_preds)
prophet_mae, prophet_rmse = evaluate_model(actual, prophet_preds)
ens_mae, ens_rmse = evaluate_model(actual, ensemble_preds)

print("\n📊 MODEL COMPARISON")
print("--------------------------------")
print(f"XGBoost  -> MAE: {xgb_mae:.2f}, RMSE: {xgb_rmse:.2f}")
print(f"ARIMA    -> MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
print(f"Prophet  -> MAE: {prophet_mae:.2f}, RMSE: {prophet_rmse:.2f}")
print(f"Ensemble -> MAE: {ens_mae:.2f}, RMSE: {ens_rmse:.2f}")

# Inventory
result = optimize_inventory(df, final_forecast)

print("\n FINAL FORECAST:", final_forecast)
print(" INVENTORY PLAN:", result)