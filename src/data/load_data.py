import pandas as pd

DATA_PATH = "C:/Users/12177/VinayMattapalli/Demand forecasting inventory/data/Walmart.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)

    print("Columns found:", df.columns)

    # Fix date
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

    # Rename
    df.rename(columns={
        'Order Date': 'Date',
        'Sales': 'Weekly_Sales'
    }, inplace=True)

    # Keep only relevant columns
    df = df[['Date', 'Weekly_Sales', 'Region', 'Category']]

    return df