import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set Matplotlib style for better visuals
plt.style.use('ggplot')

# Define the columns we want to use as extra predictors
EXTRA_REGRESSORS = ['Quantity', 'Discount', 'Profit', 'Shipping Cost']

# ————————— 1. Load & clean data —————————

def load_orders_data(filepath):
    """Load Superstore orders CSV, ensuring correct date parsing and encoding."""
    try:
        # Use low_memory=False and latin1 encoding for better compatibility
        df = pd.read_csv(filepath, parse_dates=['Order Date'], encoding='latin1', low_memory=False)
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {filepath}.")
        return pd.DataFrame()

def preprocess_orders(df):
    """
    Clean and prepare the orders dataframe:
    - Verify required columns exist
    - Ensure numeric types for 'Sales' and regressors.
    """
    # Required for any model and core data
    REQUIRED_CORE_COLS = ['Order Date', 'Sales']
    # Required for the Advanced Model
    REQUIRED_REGRESSOR_COLS = EXTRA_REGRESSORS
    REQUIRED_COLS = REQUIRED_CORE_COLS + REQUIRED_REGRESSOR_COLS
    
    if not all(col in df.columns for col in REQUIRED_COLS):
        missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
        print(f"ERROR: The data file is missing columns: {missing_cols}. Cannot proceed.")
        return pd.DataFrame()

    # Drop rows with missing order date or any required numeric column
    df = df.dropna(subset=REQUIRED_COLS)
    
    # Convert 'Sales' and all regressors to numeric
    for col in REQUIRED_CORE_COLS + REQUIRED_REGRESSOR_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df = df.dropna(subset=REQUIRED_CORE_COLS + REQUIRED_REGRESSOR_COLS)

    # Filter non-negative sales
    df = df[df['Sales'] >= 0]
    
    # Sort by date
    df = df.sort_values('Order Date').reset_index(drop=True)
    return df

# ———————— 2. Aggregate to time series ————————

def aggregate_daily_sales(df):
    """
    Aggregate the sales and all extra regressors by date (daily sum). 
    Returns a dataframe with ds / y and all regressor columns.
    """
    aggregation_dict = {'Sales': 'sum'}
    for regressor in EXTRA_REGRESSORS:
        # Summing is generally appropriate for daily aggregation of these variables
        aggregation_dict[regressor] = 'sum'
    
    daily = (
        df.groupby(df['Order Date'].dt.date)
        .agg(aggregation_dict)
        .reset_index()
        .rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    )
    daily['ds'] = pd.to_datetime(daily['ds'])
    return daily

def get_avg_regressors(train_df):
    """Calculates the mean of all extra regressors from the training data for future prediction."""
    avg_regressors = {}
    for col in EXTRA_REGRESSORS:
        avg_regressors[col] = train_df[col].mean()
    return avg_regressors

def get_holidays_df(start_date, end_date):
    """
    Creates a DataFrame of major global retail events and holidays within the data range.
    Prophet expects columns 'ds' (date) and 'holiday' (name).
    """
    years = range(start_date.year, end_date.year + 2) # Go beyond end_date for future forecast
    
    # Define globally relevant retail events and fixed-date holidays
    holidays = pd.DataFrame([
        # Key Retail Events (Often high sales)
        {'holiday': 'Black_Friday', 'ds': f'{year}-11-25'} for year in years
    ] + [
        {'holiday': 'Cyber_Monday', 'ds': f'{year}-11-28'} for year in years
    ] + [
        # Major Sales Periods
        {'holiday': 'New_Years_Day', 'ds': f'{year}-01-01'} for year in years
    ] + [
        {'holiday': 'Christmas_Eve', 'ds': f'{year}-12-24'} for year in years
    ] + [
        {'holiday': 'Christmas_Day', 'ds': f'{year}-12-25'} for year in years
    ] + [
        {'holiday': 'Boxing_Day', 'ds': f'{year}-12-26'} for year in years
    ])

    holidays['ds'] = pd.to_datetime(holidays['ds'])
    holidays = holidays.sort_values('ds').reset_index(drop=True)
    
    # Filter to only include dates within the relevant range (train + test + future)
    holidays = holidays[(holidays['ds'] >= start_date) & (holidays['ds'] <= end_date + pd.Timedelta(days=365))]
    
    return holidays

# ———————— 3. Train-test split ————————

def train_test_split_time_series(df, test_size_days=90):
    """Splits the time series data into training and testing sets."""
    df = df.sort_values('ds').reset_index(drop=True)
    cutoff = df['ds'].max() - pd.Timedelta(days=test_size_days)
    train = df[df['ds'] <= cutoff].copy()
    test = df[df['ds'] > cutoff].copy()
    return train, test

# ———————— 4. Fit Prophet & forecast ————————

def build_and_fit_prophet(train_df, holidays_df=None, regressors=None, changepoint_prior_scale=0.05):
    """Initializes and fits the Prophet model, optionally including holidays and extra regressors."""
    m = Prophet(
        holidays=holidays_df, # Pass the holidays DataFrame here
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        interval_width=0.95 
    )
    
    # Register the extra regressors if provided
    if regressors:
        for regressor in regressors:
            m.add_regressor(regressor)
        
    m.fit(train_df)
    return m

def make_validation_forecast(model, test_df):
    """Generates forecast over a known period (test set) where regressor values are known."""
    # The future dataframe is simply the test data (ds + regressors)
    if model.extra_regressors:
        future = test_df[['ds'] + list(model.extra_regressors.keys())].copy()
    else:
        future = test_df[['ds']].copy()
        
    forecast = model.predict(future)
    return forecast

def make_future_forecast(model, train_df, future_periods_days):
    """
    Generates a forecast into the unknown future.
    Uses historical averages for extra regressors.
    """
    future = model.make_future_dataframe(periods=future_periods_days, freq='D')
    
    # If using regressors, populate the future df with average historical values
    if model.extra_regressors:
        avg_regressors = get_avg_regressors(train_df)
        for regressor, avg_value in avg_regressors.items():
            # Only fill in dates that are truly in the future (Prophet's make_future_dataframe 
            # might include the last training date)
            future[regressor] = avg_value

    forecast = model.predict(future)
    return forecast

# ———————— 5. Evaluate forecast ————————

def evaluate_forecast(forecast, test_df):
    """Calculates standard time series evaluation metrics (MAE, RMSE, MSE)."""
    # Merge forecast with actuals only for the test period
    df_eval = forecast[['ds', 'yhat']].merge(test_df[['ds', 'y']], on='ds', how='inner')
    
    if df_eval.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MSE': np.nan, 'Note': 'No overlap.'}

    mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
    mse = mean_squared_error(df_eval['y'], df_eval['yhat'])
    rmse = np.sqrt(mse)
    # Return raw numbers for use in comparison logic
    return {'MAE': mae, 'RMSE': rmse, 'MSE': mse}

# ———————— 6. Plotting ————————

def plot_forecast_vs_actual(model, forecast, actual_df, title_suffix=""):
    """Plots the forecast along with actual points."""
    fig = model.plot(forecast, xlabel='Date', ylabel='Sales ($)')
    
    # Check if actual_df (test data) is provided to plot actuals
    if not actual_df.empty:
        plt.scatter(actual_df['ds'], actual_df['y'], color='red', label='Actual (Test)', s=15, alpha=0.8)
    
    plt.title(f"Sales Forecast vs Actuals {title_suffix}")
    plt.legend()
    return fig

def plot_components(model, forecast, title_suffix=""):
    """Plots the trend, yearly, weekly, holidays, and regressor components."""
    fig2 = model.plot_components(forecast)
    # Customize titles if possible (Prophet makes this difficult)
    fig2.suptitle(f"Model Components {title_suffix}", y=1.02)
    return fig2

def create_comparison_table(metrics_baseline, metrics_advanced):
    """Creates a comparison DataFrame for model metrics, formatted as strings."""
    data = {
        'Metric': ['MAE', 'RMSE'],
        'Baseline Model (No Regressors/Holidays)': [
            f"${metrics_baseline['MAE']:,.2f}", 
            f"${metrics_baseline['RMSE']:,.2f}"
        ],
        'Advanced Model (with Regressors and Holidays)': [
            f"${metrics_advanced['MAE']:,.2f}", 
            f"${metrics_advanced['RMSE']:,.2f}"
        ]
    }
    return pd.DataFrame(data).set_index('Metric')

# ———————— 7. Main flow ————————

def main(filepath='Global_Superstore.csv', test_size_days=90, future_periods=30):
    """Main execution function for running the script locally and comparing models."""
    
    df_orders = load_orders_data(filepath)
    if df_orders.empty: return
    
    df_clean = preprocess_orders(df_orders)
    if df_clean.empty: return

    df_daily = aggregate_daily_sales(df_clean)
    
    # Get the date range for holiday generation
    min_date = df_daily['ds'].min()
    max_date = df_daily['ds'].max()
    holidays_df = get_holidays_df(min_date, max_date)
    
    # Split data
    train_df, test_df = train_test_split_time_series(df_daily, test_size_days=test_size_days)
    
    print(f"\n--- Data Setup ---")
    print(f"Training Period: {train_df['ds'].min().date()} to {train_df['ds'].max().date()}")
    print(f"Testing Period: {test_df['ds'].min().date()} to {test_df['ds'].max().date()} ({len(test_df)} days)")
    print(f"Holidays Found: {len(holidays_df)} events generated.")
    
    # --- MODEL 1: BASELINE (NO REGRESSORS, NO HOLIDAYS) ---
    print("\n--- Running Baseline Model ---")
    # Note: Using the holidays_df for the baseline model as well, for consistency in the final training phase
    model_b = build_and_fit_prophet(train_df[['ds', 'y']], holidays_df=holidays_df, regressors=None)
    forecast_b = make_validation_forecast(model_b, test_df)
    metrics_b = evaluate_forecast(forecast_b, test_df)
    
    # --- MODEL 2: ADVANCED (WITH REGRESSORS AND HOLIDAYS) ---
    print("\n--- Running Advanced Model (Regressors + Holidays) ---")
    model_a = build_and_fit_prophet(train_df, holidays_df=holidays_df, regressors=EXTRA_REGRESSORS)
    forecast_a = make_validation_forecast(model_a, test_df)
    metrics_a = evaluate_forecast(forecast_a, test_df)

    # --- COMPARISON ---
    print("\n--- MODEL PERFORMANCE COMPARISON (on Test Set) ---")
    comparison_table = create_comparison_table(metrics_b, metrics_a)
    # Print the DataFrame as a string for clean console output
    print(comparison_table.to_string())
    
    # Determine the better model based on RMSE (lower is better)
    if metrics_a['RMSE'] < metrics_b['RMSE']:
        print("\n✅ The Advanced Model (with Regressors and Holidays) outperformed the Baseline Model on RMSE!")
        final_model = model_a
        final_forecast = forecast_a
    else:
        print("\n⚠️ The Baseline Model performed better or tied. Plotting Baseline Model results.")
        final_model = model_b
        final_forecast = forecast_b


    # --- FINAL FUTURE FORECAST (30 DAYS BEYOND DATA) ---
    print(f"\n--- Generating {future_periods} Day Future Forecast (Beyond Test Set) ---")
    
    # Re-fit the final model on the entire dataset (Train + Test) for the most robust future prediction
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    best_model_regressors = EXTRA_REGRESSORS if final_model.extra_regressors else None
    
    final_model_full = build_and_fit_prophet(full_df, holidays_df=holidays_df, regressors=best_model_regressors)
    
    future_forecast = make_future_forecast(final_model_full, full_df, future_periods)
    
    print(f"Forecasted Sales from {future_forecast['ds'].min().date()} to {future_forecast['ds'].max().date()}")

    # Plotting the full history validation and the future prediction
    print("\n--- Plotting Final Model Results ---")
    plot_forecast_vs_actual(final_model_full, future_forecast, pd.DataFrame(), title_suffix="(Full History and Future)")
    plot_components(final_model_full, future_forecast, title_suffix="(Full History)")
    plt.show()

if __name__ == '__main__':
    # Ensure your 'Global_Superstore.csv' is in the same directory
    main(filepath='Global_Superstore.csv')
