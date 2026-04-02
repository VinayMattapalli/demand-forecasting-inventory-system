Demand Forecasting & Inventory Optimization System

-This project is an end-to-end machine learning system designed to forecast demand and optimize inventory decisions using a combination of time-series and machine learning models. It simulates a production-style architecture with a backend API and an interactive user interface.

Overview

-The system combines statistical and machine learning approaches to improve forecasting accuracy and supports business decision-making through inventory optimization techniques.

Key capabilities include:

Demand forecasting using ARIMA, Prophet, and XGBoost
Ensemble modeling for improved prediction stability
Inventory optimization using Safety Stock, Reorder Point, and EOQ
Scenario simulation for demand fluctuations
REST API for real-time predictions
Interactive dashboard for business users

Tech Stack
Python
Pandas, NumPy
XGBoost, Prophet, Statsmodels
FastAPI
Gradio
Matplotlib

Architecture
User Interface (Gradio)
        ↓
FastAPI Backend (/forecast)
        ↓
Forecasting Models
(ARIMA + Prophet + XGBoost)
        ↓
Ensemble Layer
        ↓
Inventory Optimization Logic
        ↓
Final Output (Forecast + Inventory Plan)

Features
Multi-model demand forecasting
Ensemble-based prediction
Inventory optimization (EOQ, Safety Stock, Reorder Point)
Scenario simulation (increase/decrease demand)
Region and category-level filtering
Visualization of forecasts

How to Run
1. Install dependencies
pip install -r requirements.txt
2. Start backend API
python -m uvicorn api.fastapi_app:app --reload
3. Run the UI
python app_ui.py

Output
Forecasted demand value
Inventory plan including:
Safety Stock
Reorder Point
Economic Order Quantity (EOQ)
Scenario-based projections
Time-series visualization

Project Structure
src/
api/
app_ui.py
data/
models/
requirements.txt
README.md

Real-time data streaming (Kafka)
Deep learning models (LSTM)
Model monitoring and drift detection
