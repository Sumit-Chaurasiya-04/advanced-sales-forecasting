import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecasting_superstore import (
    load_orders_data, preprocess_orders, aggregate_daily_sales,
    train_test_split_time_series, build_and_fit_prophet, 
    make_validation_forecast, make_future_forecast,
    evaluate_forecast, plot_forecast_vs_actual, plot_components, 
    get_holidays_df, create_comparison_table, EXTRA_REGRESSORS
)

st.set_page_config(layout="wide", page_title="Advanced Sales Forecasting Dashboard")

# --- CACHED DATA PROCESSING AND MODEL RUN ---
@st.cache_data(show_spinner="Processing data and running models...")
def run_full_analysis(uploaded_file, test_days, future_days):
    """
    Loads data, preprocesses, aggregates, runs both models, and prepares outputs.
    This function is cached to prevent re-running heavy computations on every widget change.
    """
    # 1. Load & Preprocess
    df_orders = load_orders_data(uploaded_file)
    if df_orders.empty: return None, None, None, None, None
    df_clean = preprocess_orders(df_orders)
    if df_clean.empty: return None, None, None, None, None

    df_daily = aggregate_daily_sales(df_clean)
    
    # 2. Prepare Metadata
    min_date = df_daily['ds'].min()
    max_date = df_daily['ds'].max()
    holidays_df = get_holidays_df(min_date, max_date)
    
    # 3. Split Data
    train_df, test_df = train_test_split_time_series(df_daily, test_size_days=test_days)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 4. Model Training and Validation
    
    # --- MODEL 1: BASELINE (NO REGRESSORS, NO HOLIDAYS) ---
    model_b = build_and_fit_prophet(train_df[['ds', 'y']], holidays_df=holidays_df, regressors=None)
    forecast_b = make_validation_forecast(model_b, test_df)
    metrics_b = evaluate_forecast(forecast_b, test_df)

    # --- MODEL 2: ADVANCED (WITH REGRESSORS AND HOLIDAYS) ---
    model_a = build_and_fit_prophet(train_df, holidays_df=holidays_df, regressors=EXTRA_REGRESSORS)
    forecast_a = make_validation_forecast(model_a, test_df)
    metrics_a = evaluate_forecast(forecast_a, test_df)
    
    # 5. Future Forecasting (Refit on Full Data)
    # Refit both models on the full historical data for the most robust future forecast
    
    model_b_full = build_and_fit_prophet(full_df[['ds', 'y']], holidays_df=holidays_df, regressors=None)
    future_forecast_b = make_future_forecast(model_b_full, full_df, future_days)
    
    model_a_full = build_and_fit_prophet(full_df, holidays_df=holidays_df, regressors=EXTRA_REGRESSORS)
    future_forecast_a = make_future_forecast(model_a_full, full_df, future_days)

    
    return (
        df_daily, train_df, test_df, 
        (metrics_b, future_forecast_b, model_b_full), 
        (metrics_a, future_forecast_a, model_a_full)
    )

def app():
    st.title("💰 Global Superstore Sales Forecast & Analysis 🚀")
    st.markdown("Compare the **Baseline** (Seasonality only) and **Advanced** (Regressors + Holidays) forecasting models.")

    # --- SIDEBAR CONFIGURATION ---
    with st.sidebar:
        st.header("1. Data Input")
        uploaded_file = st.file_uploader(
            "Upload your 'Global_Superstore.csv' file", 
            type=["csv"],
            key="file_uploader"
        )
        
        if uploaded_file is None:
            st.warning("Please upload the orders CSV to begin.")
            st.stop()

        st.header("2. Forecast Parameters")
        test_days = st.slider(
            "Historical Test Horizon (Days for validation)", 
            30, 365, 90,
            key="test_days"
        )
        
        future_days = st.slider(
            "Future Forecast (Days beyond last data point)", 
            0, 365, 30,
            key="future_days"
        )
        
        st.markdown("---")
        st.info(f"External Regressors used: {', '.join(EXTRA_REGRESSORS)}")
        st.info("Global Retail Holidays are included in the Advanced Model.")

    # --- RUN ANALYSIS ---
    (df_daily, train_df, test_df, 
     results_b, results_a) = run_full_analysis(uploaded_file, test_days, future_days)

    if results_a is None:
        st.error("Model analysis failed. Please check the console for data loading/preprocessing errors.")
        st.stop()
        
    metrics_b, future_forecast_b, model_b_full = results_b
    metrics_a, future_forecast_a, model_a_full = results_a
    
    # --- MAIN CONTENT AREA ---
    st.subheader("Historical Sales Trend")
    st.line_chart(df_daily.set_index('ds')['y'])

    
    st.header("3. Model Performance Comparison (Validation)")
    st.markdown(f"Comparing performance over the **{len(test_df)} day** test horizon:")
    
    comparison_table_df = create_comparison_table(metrics_b, metrics_a)
    st.dataframe(comparison_table_df.style.highlight_min(axis=1, subset=['Baseline Model (No Regressors/Holidays)', 'Advanced Model (with Regressors and Holidays)']), use_container_width=True)
    
    best_model_name = "Advanced Model" if metrics_a['RMSE'] < metrics_b['RMSE'] else "Baseline Model"
    
    if metrics_a['RMSE'] < metrics_b['RMSE']:
        st.success(f"✅ The Advanced Model achieved the best RMSE (${metrics_a['RMSE']:,.2f}) and will be used for future plotting.")
    else:
        st.warning(f"⚠️ The Baseline Model performed better on RMSE (${metrics_b['RMSE']:,.2f}).")
        
    st.markdown("---")


    # --- MODEL SELECTION AND PLOTTING ---
    st.header("4. Forecast Visualization")
    
    plot_model_type = st.radio(
        "Select Model for Detailed Plotting:",
        ('Advanced Model (Regressors + Holidays)', 'Baseline Model (Seasonality Only)'),
        key="plot_model_select"
    )
    
    if plot_model_type.startswith('Advanced'):
        current_model = model_a_full
        current_forecast = future_forecast_a
    else:
        current_model = model_b_full
        current_forecast = future_forecast_b
    
    # Plot 1: Forecast vs Actuals
    st.subheader(f"Forecast vs Actuals ({plot_model_type})")
    
    # Show the full history forecast including the future prediction
    fig_forecast = plot_forecast_vs_actual(
        current_model, 
        current_forecast, 
        test_df, 
        title_suffix=f"({future_days} Day Future)"
    )
    st.pyplot(fig_forecast, use_container_width=True)

    
    # Plot 2: Components (Trend, Seasonality, Regressors, Holidays)
    st.subheader(f"Decomposition Components ({plot_model_type})")
    fig_components = plot_components(current_model, current_forecast, title_suffix=plot_model_type)
    st.pyplot(fig_components, use_container_width=True)


if __name__ == "__main__":
    app()
