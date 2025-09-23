# Required libraries
import pandas as pd
import numpy as np
import os
import copy
from solver import *



# --- GENERAL PARAMETERS ---

# Path to "data" folder
data_path = os.path.join(os.path.dirname(__file__), "data")

# Path to "results" folder
results_path = os.path.join(os.path.dirname(__file__), "results", "results_mixture")

# Set the variable save_figures=True to save the result images
save_figures = True



# ------ DATA IMPORT ------
sp500_companies, tickers, data_stocks, market_caps_df_2020, market_caps_df_2021, market_caps_dict = import_data(data_path)


# ------ PREPROCESSING HISTORICAL STOCKS DATA ------
sp500_companies, tickers, data_stocks = data_preprocessing(sp500_companies, tickers, data_stocks)



# ------ MODELS TRAINING 2020 ------
# --- CALCULATE RETURNS, MARKET CAPS, COVARIANCE MATRIX AND CORRELATION MATRIX ---

# Define the time interval
interval_2020 = ('2019-12-31', '2020-12-31')  # Interval 1 - 2020

# Create a new filtered dictionary for the interval
data_stocks_interval_2020 = filter_by_date_range(data_stocks, *interval_2020)


# To calculate the returns for each time interval, start from the second value and loop
# over the data_stocks_interval_1 dictionary. For each DataFrame associated 
# with the keys (companies), add a new column for the calculated returns.

# Loop over each dictionary and over each ticker in the dictionary
for ticker in data_stocks_interval_2020:
    data_stocks_interval_2020[ticker] = data_stocks_interval_2020[ticker].copy()
    data_stocks_interval_2020[ticker]['Return'] = data_stocks_interval_2020[ticker]['Close'].pct_change()


# Build a new data structure for the stock data over the 2020 interval
new_data_stocks_interval_2020 = create_new_data_structure(data_stocks_interval_2020)

# Calculate the initial weights (w0) of the stock market caps in 2020
total_market_caps_2020 = market_caps_df_2020['Market Cap'].sum()
w0_2020 = {azienda: market_caps_df_2020.loc[azienda, 'Market Cap'] / total_market_caps_2020 for azienda in market_caps_df_2020.index}

# Build the returns matrix for all stocks in 2020
returns_matrix_2020 = []

for ticker in tickers:
    df = data_stocks_interval_2020[ticker]
    returns_matrix_2020.append(df['Return'][1:].values)  # each vector has dimension T

# Convert the list of vectors into a matrix of dimension T x n (observations x assets)
returns_matrix_2020 = np.array(returns_matrix_2020).T  

# Convert w0 into a numpy array, ordered according to the tickers
weights_0 = np.array([w0_2020[ticker] for ticker in tickers])

# Compute epsilon_b: benchmark returns as the weighted sum of asset returns
epsilon_b = returns_matrix_2020 @ weights_0  # dimensione: T x 1

# Number of distributions (components) in the Gaussian mixture
d = 3

# Build the data matrix X of stock returns (dimensions: T x n)
df_X = pd.DataFrame({stock: values['Return'][1:] for stock, values in new_data_stocks_interval_2020.items()})
X = df_X.values

# Add the benchmark returns as an additional dimension to X
eps_benchmark = epsilon_b.reshape(-1, 1) # column vector of dimension T x 1
X_new = np.concatenate([X, eps_benchmark], axis=1) # final dimension: T x (n+1)

# Perform k-means initialization for the EM algorithm parameters
mu_init, lambda_init, cov_init = kmeans_initialization(X_new, d)

# Run the EM Iteration Process (EMIP) for the Gaussian mixture
mu, cov, lamb = sklearn_function(X_new, new_data_stocks_interval_2020, mu_init, cov_init, lambda_init, d)

# Solve optimization models and store results for different portfolio specifications
# Models: 1 - EIT, 2 - DEIT
results_model_1, results_model_2 = save_model_results_gurobi(
    lamb, mu, cov, market_caps_df_2020, w0_2020, new_data_stocks_interval_2020    
    )


# ------ OUT-OF-SAMPLE EVALUATION OVER MULTIPLE PERIODS (2021–2023) ------

# Define the initial and final interval
start_interval = ('2019-12-31', '2020-12-31')
end_interval = ('2022-12-31', '2023-12-31')

# Convert dates to datetime format
start_date = pd.to_datetime(start_interval[0])
end_date = pd.to_datetime(end_interval[0])
delta = pd.DateOffset(months=3)  # Monthly step

# intervals = annual rolling windows (kept for compatibility with older functions)
intervals = []
while start_date <= end_date:
    end_date_interval = start_date + pd.DateOffset(years=1) 
    intervals.append((start_date.strftime('%Y-%m-%d'), end_date_interval.strftime('%Y-%m-%d')))
    start_date += delta  # 3 Monthly shift

# Delete first element('2019-12-31', '2020-12-31')
del intervals[0]

# new_intervals = last 3 months of each annual window (used for actual out-of-sample evaluation)
new_intervals = []
for _, end_str in intervals:
    end = pd.to_datetime(end_str)
    start = end - pd.DateOffset(months=3)
    new_intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
    
# Dictionaries to store results for each out-of-sample interval
results_model_1_roll_out={}
results_model_2_roll_out={} 
tracking_error_dict_1={}
tracking_error_dict_2={}

# Temporary containers for model results
temp_res1 = {}
temp_res2 = {}


    
# Copy model results for each evaluation interval
for h in new_intervals:
    temp_res1[h] = copy.deepcopy(results_model_1)
    temp_res2[h] = copy.deepcopy(results_model_2)

# List to collect portfolio performance metrics
data_list_out=[]

# --- Evaluate portfolios on each out-of-sample interval ---
for interval_out, market_caps_out in market_caps_dict.items():
    # Define start and end of the 3-month evaluation window
    end = pd.to_datetime(interval_out[1])
    start = end - pd.DateOffset(months=3)
    interval_out = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        
    # Filter stock data for the current interval
    data_stocks_interval_roll_out = filter_by_date_range(data_stocks, *interval_out)

    # Compute percentage returns for each stock in the filtered dictionary
    for ticker in data_stocks_interval_roll_out:
        data_stocks_interval_roll_out[ticker] = data_stocks_interval_roll_out[ticker].copy()
        data_stocks_interval_roll_out[ticker]['Return'] = data_stocks_interval_roll_out[ticker]['Close'].pct_change()
    
    # Restructure the data for further analysis
    new_data_stocks_interval_roll_out = create_new_data_structure(data_stocks_interval_roll_out)
    
    # Compute initial weights (w0) based on stock market capitalization
    total_market_caps_roll_out = market_caps_out['Market Cap'].sum()
    w0_roll_out = {azienda: market_caps_out.loc[azienda, 'Market Cap'] / total_market_caps_roll_out for azienda in market_caps_out.index}

    # Calculate the covariance matrix
    covariance_matrix_roll_out = calculate_covariance_matrix(new_data_stocks_interval_roll_out)

    # --- Calculate portfolio returns ---
    results_model_1_roll_out[interval_out], results_model_2_roll_out[interval_out], index_mean_returns_roll_out, index_return_roll_out = calculate_portfolio_return_mixture(temp_res1[interval_out], temp_res2[interval_out], new_data_stocks_interval_roll_out, w0_roll_out)
    
    # --- Calculate portfolio variances ---
    results_model_1_roll_out[interval_out], results_model_2_roll_out[interval_out], index_variance_roll_out = calculate_portfolio_variance_mixture(covariance_matrix_roll_out, w0_roll_out, temp_res1[interval_out], temp_res2[interval_out])
        
    # --- Calculate portfolio Sharpe Ratios ---
    results_model_1_roll_out[interval_out], results_model_2_roll_out[interval_out], SR_index_roll_out = calculate_sharpe_ratios_mixture(temp_res1[interval_out], temp_res2[interval_out], index_return_roll_out, index_variance_roll_out)
    
    # Collect results into a row
    row = {"Interval": interval_out, "index_return": index_return_roll_out, "index_variance": index_variance_roll_out,"SR_index": SR_index_roll_out}
    data_list_out.append(row)  
    
    # Calculate Tracking Error
    track_error_1, track_error_2 = calculate_tracking_error_out_mixture(interval_out, covariance_matrix_roll_out, w0_roll_out, results_model_1_roll_out, results_model_2_roll_out) 
    tracking_error_dict_1[interval_out]=track_error_1
    tracking_error_dict_2[interval_out]=track_error_2


# Convert the list into a DataFrame
index_return_var_out = pd.DataFrame(data_list_out)

# ------ OUT-OF-SAMPLE PERFORMANCE EVALUATION RESULTS ------

# Plot portfolio returns
plot_portfolio_return_out_mixture(new_intervals, index_return_var_out, results_model_1_roll_out, results_model_2_roll_out, save_figures, results_path)

# Plot portfolio variances
plot_portfolio_variance_out_mixture(new_intervals, index_return_var_out, results_model_1_roll_out, results_model_2_roll_out, save_figures, results_path)

# Plot portfolio Sharpe Ratios
plot_portfolio_sharpe_ratios_out_mixture(new_intervals, index_return_var_out, results_model_1_roll_out, results_model_2_roll_out, save_figures, results_path)

# Tracking Ratios (out-of-sample evaluation)
tracking_ratio_dict_1, tracking_ratio_dict_2 = calculate_tracking_ratio_out_mixture(new_intervals, market_caps_dict, market_caps_df_2020, results_model_1, results_model_1_roll_out, results_model_2, results_model_2_roll_out)
plot_tracking_ratio_out_mixture(new_intervals, tracking_ratio_dict_1, tracking_ratio_dict_2, save_figures, results_path)

# Tracking Errors (out-of-sample evaluation)
plot_tracking_error_out_mixture(new_intervals, tracking_error_dict_1, tracking_error_dict_2, save_figures, results_path)
