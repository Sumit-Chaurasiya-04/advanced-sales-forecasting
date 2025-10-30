🚀 Advanced Sales Forecasting with Prophet (Superstore Analysis)

This project showcases a robust time-series forecasting pipeline using the Prophet library, built by Facebook. It is specifically designed for sales data analysis, featuring a direct comparison between a simple baseline model and an advanced model that incorporates External Regressors (like Quantity, Profit, etc.) to significantly improve predictive accuracy.

The project is fully containerized and features an interactive web dashboard built with Streamlit, allowing users to upload data and test different forecast horizons in real-time.

✨ Features

Dual-Model Comparison: Automatically compares a Baseline model (Trend + Seasonality only) against an Advanced model (Trend + Seasonality + Regressors).

External Regressors: Utilizes key transactional metrics (Quantity, Discount, Profit, Shipping Cost) as external regressors to capture non-seasonal impacts on sales.

Interactive Streamlit Dashboard: A user-friendly web application for data upload, parameter tuning, and dynamic visualization of results.

Future Forecasting: Includes logic to forecast sales into the unknown future by making reasonable assumptions (using historical averages) for the required external regressors.

Detailed Metrics: Provides clear validation metrics (MAE, RMSE, MSE) to quantify model performance.

🛠️ Setup and Installation

Prerequisites

You need Python 3.8+ installed on your system.

1. Clone the repository

git clone [https://github.com/your-username/your-superstore-project.git](https://github.com/your-username/your-superstore-project.git)
cd your-superstore-project


2. Install Dependencies

Install all required Python packages using the provided requirements.txt file.

pip install -r requirements.txt


3. Data Preparation

Place your data file, Global_Superstore.csv, into the root directory of the project.

Note: Your CSV must contain, at a minimum, the columns Order Date, Sales, Quantity, Discount, Profit, and Shipping Cost.

🚀 How to Run the Project

You have two main ways to interact with the project:

Option A: Run the Interactive Dashboard (Recommended)

Start the Streamlit application to access the full interactive interface in your browser.

streamlit run dashboard.py


This will automatically open the dashboard, allowing you to upload your CSV file and adjust the parameters on the sidebar.

Option B: Run the Standalone Comparison Script

Execute the core Python script to run the full two-model comparison locally, outputting the metrics table and showing all plots.

python forecasting_superstore.py


This script will print the comparison table to the console and display the generated Matplotlib charts for the best-performing model.