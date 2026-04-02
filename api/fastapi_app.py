from fastapi import FastAPI
import traceback

from src.data.load_data import load_data
from src.models.predict_models import predict_all
from src.ensemble.ensemble import ensemble_forecast
from src.inventory.inventory_model import optimize_inventory

app = FastAPI()


@app.get("/forecast")
def get_forecast():
    try:
        df = load_data()

        xgb, arima, prophet = predict_all()

        final = ensemble_forecast(xgb, arima, prophet)

        inventory = optimize_inventory(df, final)

        return {
            "xgb": float(xgb),
            "arima": float(arima),
            "prophet": float(prophet),
            "final_forecast": float(final),
            "inventory": inventory
        }

    except Exception as e:
        return {
            "error": str(e),
            "trace": traceback.format_exc()
        }