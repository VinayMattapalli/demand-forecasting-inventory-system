import gradio as gr
import matplotlib.pyplot as plt
import requests

from src.data.load_data import load_data
from src.features.build_features import build_features
from src.inventory.inventory_model import optimize_inventory, simulate_demand_change


def run_system(region, category, change_percent):

    df = load_data()

    if region != "All":
        df = df[df['Region'] == region]

    if category != "All":
        df = df[df['Category'] == category]

    df = df.groupby('Date').agg({'Weekly_Sales': 'sum'}).reset_index()
    df = build_features(df)

    response = requests.get("http://127.0.0.1:8000/forecast")
    data = response.json()

    print("API RESPONSE:", data)

    if "error" in data:
        return f"API Error: {data['error']}", "", "", None

    final = data.get("final_forecast", 0)
    inventory = data.get("inventory", {})

    scenario = simulate_demand_change(df, final, change_percent)

    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Weekly_Sales'], label='Actual')
    plt.axhline(y=final, color='r', linestyle='--', label='Forecast')
    plt.legend()

    return (
        f"Final Forecast: {final:.2f}",
        f"Inventory: {inventory}",
        f"Scenario (+{change_percent}%): {scenario}",
        plt
    )


df = load_data()

regions = ["All"] + list(df['Region'].dropna().unique())
categories = ["All"] + list(df['Category'].dropna().unique())


interface = gr.Interface(
    fn=run_system,
    inputs=[
        gr.Dropdown(regions, label="Region"),
        gr.Dropdown(categories, label="Category"),
        gr.Slider(-50, 50, value=20, label="Demand Change %")
    ],
    outputs=["text", "text", "text", "plot"],
    title=" Demand Forecasting System (FAANG Level)"
)

interface.launch()