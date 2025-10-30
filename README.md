💰 Advanced Sales Forecasting: Global Superstore Analysis 🚀

Repository Link: https://github.com/Sumit-Chaurasiya-04/advanced-sales-forecasting

Project Overview

This project implements a sophisticated time series analysis pipeline using the Prophet library for sales forecasting on the Global Superstore dataset. It is designed to rigorously compare a simple baseline model against an advanced model incorporating External Regressors and Global Holidays to achieve superior accuracy and provide actionable business insights through an interactive Streamlit dashboard.

Key Features

Feature

Description

Advanced Technique

Model Comparison

Automatically trains and validates a simple Baseline Model (Seasonality only) against the Advanced Model.

Direct RMSE and MAE comparison on a dedicated test set.

External Regressors

Includes key operational metrics (Quantity, Discount, Profit, Shipping Cost) as independent variables.

Multivariate Time Series Analysis via Prophet's add_regressor.

Global Holidays

Accounts for predictable sales spikes around major retail events (Black Friday, Christmas).

Custom Holidays Dataframe to handle non-seasonal anomalies.

Interactive Dashboard

A polished web application built with Streamlit for non-technical users to upload data and adjust forecast horizons.

Full MVT (Model-View-Template) separation for clear project structure.

Future Prediction

Models are retrained on the full dataset and used to forecast sales for future periods (e.g., 30 days beyond the last historical date).

Realistic Production-ready forecasting setup.

🛠️ Technology Stack

Core Libraries: pandas, numpy, matplotlib, seaborn

Forecasting: Prophet (developed by Meta)

Web Application: Streamlit

Environment Management: pip

⚙️ Setup and Installation

Follow these steps to get a local copy of the project running on your machine.

1. Clone the Repository

Open your terminal or Git Bash and run:

git clone [https://github.com/Sumit-Chaurasiya-04/advanced-sales-forecasting.git](https://github.com/Sumit-Chaurasiya-04/advanced-sales-forecasting.git)
cd advanced-sales-forecasting


2. Prepare the Data

Ensure your Global_Superstore.csv file is placed directly inside the project root directory. The file must contain the following columns for successful execution:

Order Date

Sales

Quantity

Discount

Profit

Shipping Cost

(Note: The .gitignore file excludes this large data file from the repository, so you must provide it locally.)

3. Install Dependencies

It is highly recommended to use a virtual environment.

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt


🚀 Running the Project

You have two ways to run the analysis: the command-line script for comparison, and the Streamlit app for interactive use.

A. Run the Core Analysis Script

This script trains both models, prints the performance comparison, and displays the final plots.

python forecasting_superstore.py


B. Run the Interactive Dashboard

This command launches the Streamlit web application in your browser.

streamlit run dashboard.py


A browser window will open, allowing you to upload your Global_Superstore.csv file and interactively adjust the forecast parameters.

📈 Model Methodology (The Advanced Difference)

The core logic trains two models to demonstrate the value of advanced feature engineering:

Baseline Model:


$$Y_{t} = \text{Trend} + \text{Seasonality} + \text{Error}$$


Uses only time features (weekly, yearly seasonality) to model sales.

Advanced Model:


$$Y_{t} = \text{Trend} + \text{Seasonality} + \text{Holidays} + \beta_1 \cdot \text{Quantity} + \beta_2 \cdot \text{Discount} + \beta_3 \cdot \text{Profit} + \beta_4 \cdot \text{Shipping Cost} + \text{Error}$$


This model explicitly models the impact of known events (holidays) and business decisions (regressors), leading to a significant reduction in forecast error.

Next Steps (Future Enhancements)

To take this project to the next level of analytical rigor, the following enhancement is planned:

Automated Hyperparameter Tuning: Implement a grid search or cross-validation approach to automatically find the optimal changepoint_prior_scale and seasonality_mode for the Prophet models, ensuring the statistically best possible forecast.
