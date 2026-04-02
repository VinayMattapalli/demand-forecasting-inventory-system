Demand Forecasting & Inventory Optimization System

this project is a full end-to-end machine learning system that combines time-series forecasting with business decision-making.


 Features
Hybrid Forecasting Model:
ARIMA (time series trends)
Prophet (seasonality)
XGBoost (ML features)
Ensemble Learning for improved accuracy
Inventory Optimization:
Safety Stock
Reorder Point
EOQ
Scenario Simulation:
Demand increase/decrease analysis
Interactive UI:
Gradio dashboard
Region & Category filters
Production API:
FastAPI backend
Real-time predictions

 Tech Stack
Python
Pandas, NumPy
XGBoost, Prophet, Statsmodels
FastAPI
Gradio
Matplotlib

Architecture
UI (Gradio)
    ↓
FastAPI Backend
    ↓
ML Models (ARIMA + Prophet + XGBoost)
    ↓
Ensemble
    ↓
Inventory Optimization

How to Run
1. Install dependencies
-pip install -r requirements.txt
2. Start API
-python -m uvicorn api.fastapi_app:app --reload
3. Run UI
-python app_ui.py

Output
-Demand Forecast
-Inventory Plan
-Scenario Simulation
-Visualization Graph
