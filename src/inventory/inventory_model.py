import numpy as np


def optimize_inventory(df, forecast):

    std_dev = df['Weekly_Sales'].std()
    lead_time = 2
    z = 1.65

    safety_stock = z * std_dev * np.sqrt(lead_time)
    reorder_point = (forecast * lead_time) + safety_stock

    annual_demand = df['Weekly_Sales'].mean() * 52
    ordering_cost = 50
    holding_cost = 2

    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

    return {
        "forecast": float(forecast),
        "safety_stock": float(safety_stock),
        "reorder_point": float(reorder_point),
        "eoq": float(eoq)
    }


# Simulate demand change scenario
def simulate_demand_change(df, forecast, change_percent):

    new_forecast = forecast * (1 + change_percent / 100)

    result = optimize_inventory(df, new_forecast)

    return {
        "new_forecast": float(new_forecast),
        "inventory_plan": result
    }