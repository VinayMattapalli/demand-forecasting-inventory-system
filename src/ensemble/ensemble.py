def ensemble_forecast(xgb, arima, prophet):

    # weighted ensemble
    final = (0.5 * xgb) + (0.2 * arima) + (0.3 * prophet)

    return final