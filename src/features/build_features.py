def build_features(df):

    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month

    # Extract week number using isocalendar, which returns a DataFrame with 'year', 'week', and 'day' columns
    df['week'] = df['Date'].dt.isocalendar().week
    df['week'] = df['week'].fillna(0).astype(int)

    # Handle any remaining NaN values
    df = df.fillna(0)

    return df