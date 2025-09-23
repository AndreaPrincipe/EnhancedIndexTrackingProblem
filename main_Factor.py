# Required libraries
import pandas as pd
import numpy as np
import os
import copy
from solver import *
from itertools import islice


# --- GENERAL PARAMETERS ---

# Path to "data" folder
data_path = os.path.join(os.path.dirname(__file__), "data")

# Path to "results" folder
results_path = os.path.join(os.path.dirname(__file__), "results", "results_factor")

# Define the q values for which to obtain results. It's the number of clusters
q_values = [11, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Define the q values useful for Out-of-samples dynamic test
q_values_roll=[30, 40, 50]

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


# Calculate new structure
new_data_stocks_interval_2020 = create_new_data_structure(data_stocks_interval_2020)

# Calculate the covariance matrix
covariance_matrix_2020 = calculate_covariance_matrix(new_data_stocks_interval_2020)

# --- RESULTS OF MODELS FOR DIFFERENT PORTFOLIO SIZES --- 

# Calculate the initial weights (w0) of the stock market caps in 2020
total_market_caps_2020 = market_caps_df_2020['Market Cap'].sum()
w0_2020 = {azienda: market_caps_df_2020.loc[azienda, 'Market Cap'] / total_market_caps_2020 for azienda in market_caps_df_2020.index}

#fama-french 5 factor model
start_date_1, end_date_1 = interval_2020
smb, hml, RF, mkt_RF, RMW, CMA = five_factor_FF(start_date_1, end_date_1, data_path)

#data used for the basic model
#define mu
keys = list(new_data_stocks_interval_2020.keys())

# Creiamo un vettore vuoto di lunghezza n
n = len(new_data_stocks_interval_2020)
mu = np.zeros(n)

# Ciclo per inserire i valori nella posizione corretta
for i, key in enumerate(keys):
    mu[i] = new_data_stocks_interval_2020[key]["Mean Return"].iloc[0]

Cov_sqrt = sqrt_matrix(covariance_matrix_2020)

#we need the standard deviation for the parameters sigma and TE
std_dev = np.sqrt(np.diag(covariance_matrix_2020))
p = 95                                          
p_std = np.percentile(std_dev, p)

sigma = 6*p_std
TE = 3*p_std

#call the function that give us the estimates for the robust model
factors_matrix_3 = np.column_stack((mkt_RF, RF, smb, hml))  # (253 x 4)
mu_0, V0, rho, gamma_up, gamma_down, G, F, d_up = estimates_parameters_general(factors_matrix_3, new_data_stocks_interval_2020)

factors_matrix_5 = np.column_stack((mkt_RF, RF, smb, hml, RMW, CMA))  # (253 x 6)
mu_0_5, V0_5, rho_5, gamma_up_5, gamma_down_5, G_5, F_5, d_up_5 = estimates_parameters_general(factors_matrix_5, new_data_stocks_interval_2020)

# Generate outputs for the 3 models for varying portfolio size q and save the results in results_model_i, for i=1,2,3.

#1-nominal model, 2-fama-french 3 factor model, 3-fama-french 5 factor model
results_model_1, results_model_2, results_model_3, time_m_1, time_m_2, time_m_3 = save_model_results_mosek(
    q_values, mu, Cov_sqrt, w0_2020, sigma, TE, market_caps_df_2020, mu_0, gamma_down, F, V0, rho, G, d_up, mu_0_5, gamma_down_5, F_5, V0_5, rho_5, G_5, d_up_5
    )


# Make a copy of the results_model_i, for i=1,2,3,4 useful for testing models on out-of-sample data
results_model_1_out = copy.deepcopy(results_model_1)
results_model_2_out = copy.deepcopy(results_model_2)
results_model_3_out = copy.deepcopy(results_model_3)


# ------ COMPUTATIONAL RESULTS OF MODELS TRAINING ------

# --- Comparison plot of the sector weight difference norms between the estimated portfolios and the S&P500 target ---
plot_norm_differences(q_values, results_model_1, results_model_2, results_model_3, save_figures, results_path)

# Plot sector diversification process
#values={11,30,100}
#combined_results = analyze_sector_proportions(values, sp500_companies, results_model_1, results_model_2, results_model_3, results_model_4, unique_sectors_sorted, save_figures, results_path)

# --- Comparison of performances via the optimal value of the objective function ---
plot_objective_values(q_values, results_model_1, results_model_2, results_model_3, save_figures, results_path)

# --- Comparison of portfolio returns ---
results_model_1, results_model_2, results_model_3, index_mean_returns_2020, index_return_2020 = calculate_portfolio_return(results_model_1, results_model_2, results_model_3, new_data_stocks_interval_2020, w0_2020)
# Plot the graph
plot_portfolio_return_2020 = plot_portfolio_return_comparison(q_values, index_return_2020, results_model_1, results_model_2, results_model_3, save_figures, results_path)
   

# --- Comparison of portfolio variances ---
results_model_1, results_model_2, results_model_3, index_variance_2020 = calculate_portfolio_variance(covariance_matrix_2020, w0_2020, results_model_1, results_model_2, results_model_3, q_values)
# Plot the graph
plot_portfolio_variance_2020 = plot_portfolio_variance_comparison(results_model_1, results_model_2, results_model_3, index_variance_2020, q_values, save_figures, results_path)
        

# --- Comparison of portfolio Sharpe Ratios ---
results_model_1, results_model_2, results_model_3, SR_index_2020 = calculate_sharpe_ratios(results_model_1, results_model_2, results_model_3, index_return_2020, index_variance_2020, q_values)
# Plot the graph
plot_portfolio_sharpe_ratios_2020 = plot_sharpe_ratios_comparison(results_model_1, results_model_2, results_model_3, SR_index_2020, q_values, save_figures, results_path)



# ------ STATIC TESTING OUT-OF-SAMPLE 2021 ------

# Define the time interval
interval_2021 = ('2020-12-31', '2021-12-31')  # Interval 2 - 2021

# Create a new filtered dictionary for the time range
data_stocks_interval_2021 = filter_by_date_range(data_stocks, *interval_2021)

# Iterate through each dictionary
for ticker in data_stocks_interval_2021:
    data_stocks_interval_2021[ticker] = data_stocks_interval_2021[ticker].copy()  # Assicura una copia indipendente
    data_stocks_interval_2021[ticker]['Return'] = data_stocks_interval_2021[ticker]['Close'].pct_change()


# Calculate the new structure
new_data_stocks_interval_2021 = create_new_data_structure(data_stocks_interval_2021)

# Calculate the covariance matrix
covariance_matrix_2021 = calculate_covariance_matrix(new_data_stocks_interval_2021)

# Calculate the initial weights (w0_2) of the stock capitalizations as of 2021
total_market_caps_2021 = market_caps_df_2021['Market Cap'].sum()
w0_2021 = {azienda: market_caps_df_2021.loc[azienda, 'Market Cap'] / total_market_caps_2021 for azienda in market_caps_df_2021.index}


# ------ COMPUTATIONAL RESULTS OF MODELS TESTING 2021 OUT-OF-SAMPLE ------

# Comparison of portfolio returns
results_model_1_out, results_model_2_out, results_model_3_out, index_mean_returns_2021, index_return_2021 = calculate_portfolio_return(results_model_1_out, results_model_2_out, results_model_3_out, new_data_stocks_interval_2021, w0_2021)
plot_portfolio_return_out = plot_portfolio_return_comparison(q_values, index_return_2021, results_model_1_out, results_model_2_out, results_model_3_out, save_figures, results_path)

# Comparison of portfolio variances
results_model_1_out, results_model_2_out, results_model_3_out, index_variance_2021 = calculate_portfolio_variance(covariance_matrix_2021, w0_2021, results_model_1_out, results_model_2_out, results_model_3_out, q_values)
plot_portfolio_variance_out = plot_portfolio_variance_comparison(results_model_1_out, results_model_2_out, results_model_3_out, index_variance_2021, q_values, save_figures, results_path)

# Comparison of portfolio sharpe ratioS
results_model_1_out, results_model_2_out, results_model_3_out, SR_index_2021 = calculate_sharpe_ratios(results_model_1_out, results_model_2_out, results_model_3_out, index_return_2021, index_variance_2021, q_values)
plot_portfolio_sharpe_ratios_out = plot_sharpe_ratios_comparison(results_model_1_out, results_model_2_out, results_model_3_out, SR_index_2021, q_values, save_figures, results_path)



# ------ GRAPHIC COMPARISON OF MODELS TESTING WITH IN-SAMPLE (2020) AND OUT-OF-SAMPLE (2021) DATA ------

# Compare portfolio return
plot_portfolio_return_in_out = figures_merge(plot_portfolio_return_2020, plot_portfolio_return_out, save_figures, results_path, file_name="return_comparison_in_out.png")

# Compare portfolio variance
plot_portfolio_variance_in_out = figures_merge(plot_portfolio_variance_2020, plot_portfolio_variance_out, save_figures, results_path, file_name="variance_comparison_in_out.png")

# Compare portfolio Sharpe ratios
plot_portfolio_sharpe_ratios_in_out = figures_merge(plot_portfolio_sharpe_ratios_2020, plot_portfolio_sharpe_ratios_out, save_figures, results_path, file_name="sharpe_ratios_comparison_in_out.png")



# ------ ANALYTICAL COMPARISON OF MODELS TESTING WITH IN-SAMPLE (2020) AND OUT-OF-SAMPLE (2021) DATA ------

# Compare portfolio returns
return_models_in_out = return_comparison_in_out(q_values, results_model_1, results_model_1_out, results_model_2, results_model_2_out, results_model_3, results_model_3_out)

# Compare portfolio variances
variance_models_in_out = variance_comparison_in_out(q_values, results_model_1, results_model_1_out, results_model_2, results_model_2_out, results_model_3, results_model_3_out)

# Compare portfolio Sharpe ratios
sharpe_ratios_models_in_out = sharpe_ratios_comparison_in_out(q_values, results_model_1, results_model_1_out, results_model_2, results_model_2_out, results_model_3, results_model_3_out)

# Tracking Ratios in-sample and out-of-sample
tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3 = calculate_tracking_ratio(results_model_1, results_model_1_out, results_model_2, results_model_2_out, results_model_3, results_model_3_out, market_caps_df_2020, market_caps_df_2021, total_market_caps_2020, total_market_caps_2021, q_values)
plot_portfolio_tracking_ratio_in_out = plot_tracking_ratio(tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3, q_values, save_figures, results_path)

# Tracking Errors in-sample and out-of-sample
tracking_error_model_1, tracking_error_model_2, tracking_error_model_3 = calculate_tracking_error(q_values, w0_2021, covariance_matrix_2021, results_model_1_out, results_model_2_out, results_model_3_out)
plot_tracking_error(tracking_error_model_1, tracking_error_model_2, tracking_error_model_3, q_values, save_figures, results_path)



# ------ TRAINING AND DYNAMIC TESTING OUT-OF-SAMPLE ROLLING WINDOWS (2020-2023) ------

# Define the initial and final interval
start_interval = ('2019-12-31', '2020-12-31')
end_interval = ('2022-12-31', '2023-12-31')

# Convert dates to datetime format
start_date = pd.to_datetime(start_interval[0])
end_date = pd.to_datetime(end_interval[0])
delta = pd.DateOffset(months=3)  # Monthly step

# Generate all intervals by shifting the period with a monthly step
intervals = []
while start_date <= end_date:
    end_date_interval = start_date + pd.DateOffset(years=1) #spostamento temporale indicato con DataOffset
    intervals.append((start_date.strftime('%Y-%m-%d'), end_date_interval.strftime('%Y-%m-%d')))  #convertiamo in stringhe per usare le funzioni precedenti senza problemi
    start_date += delta  # 3 Monthly shift

# Delete first element('2019-12-31', '2020-12-31')
del intervals[0]

# Compute the 3 models on out-of-samples data with rolling windows
results_model_1_roll, results_model_2_roll, results_model_3_roll = perform_rolling_analysis(market_caps_dict, data_stocks, sp500_companies, q_values_roll, data_path)

# Make a copy of the results_model_i, for i=1,2,3, useful for testing models on out-of-sample data
results_model_1_roll_copy = copy.deepcopy(results_model_1_roll)
results_model_2_roll_copy = copy.deepcopy(results_model_2_roll)
results_model_3_roll_copy = copy.deepcopy(results_model_3_roll)



# ------ CALCULATE DATA FOR IN-SAMPLE TIME INTERVALS ------

for interval, market_caps in market_caps_dict.items():
       
    # Create a new filtered dictionary for the interval
    data_stocks_interval_roll = filter_by_date_range(data_stocks, *interval)
        
    for ticker in data_stocks_interval_roll:
        data_stocks_interval_roll[ticker] = data_stocks_interval_roll[ticker].copy()
        data_stocks_interval_roll[ticker]['Return'] = data_stocks_interval_roll[ticker]['Close'].pct_change()

    
    # Restructure the data for further analysis
    new_data_stocks_interval_roll = create_new_data_structure(data_stocks_interval_roll)

    # Calculate the covariance matrix
    covariance_matrix_roll = calculate_covariance_matrix(new_data_stocks_interval_roll)

    # Compute initial weights (w0) based on stock market capitalization
    total_market_caps_roll = market_caps['Market Cap'].sum()
    w0_roll = {azienda: market_caps.loc[azienda, 'Market Cap'] / total_market_caps_roll for azienda in market_caps.index}
    
    # --- Calculate portfolio returns ---
    results_model_1_roll[interval], results_model_2_roll[interval], results_model_3_roll[interval], index_mean_returns_roll, index_return_roll = calculate_portfolio_return(results_model_1_roll[interval], results_model_2_roll[interval], results_model_3_roll[interval], new_data_stocks_interval_roll, w0_roll)
    
    # --- Calculate portfolio variances ---
    results_model_1_roll[interval], results_model_2_roll[interval], results_model_3_roll[interval], index_variance_roll = calculate_portfolio_variance(covariance_matrix_roll, w0_roll, results_model_1_roll[interval], results_model_2_roll[interval], results_model_3_roll[interval], q_values_roll)
        
    # --- Calculate portfolio Sharpe Ratios ---
    results_model_1_roll[interval], results_model_2_roll[interval], results_model_3_roll[interval], SR_index_roll = calculate_sharpe_ratios(results_model_1_roll[interval], results_model_2_roll[interval], results_model_3_roll[interval], index_return_roll, index_variance_roll, q_values_roll)
 


# ------ CALCULATE DATA FOR OUT-OF-SAMPLE TIME INTERVALS ------

results_model_1_roll_out={}
results_model_2_roll_out={}
results_model_3_roll_out={}

tracking_error_dict_1={}
tracking_error_dict_2={}
tracking_error_dict_3={}

intervals_out=intervals[4:]
data_list_out=[]

interval=intervals[0]
i=0
for interval_out, market_caps_out in islice(market_caps_dict.items(), 4, None):
    
    # Create a new filtered dictionary for the interval
    data_stocks_interval_roll_out = filter_by_date_range(data_stocks, *interval_out)

    # Compute percentage returns for each stock in the filtered dictionary
    for ticker in data_stocks_interval_roll_out:
        data_stocks_interval_roll_out[ticker] = data_stocks_interval_roll_out[ticker].copy()
        data_stocks_interval_roll_out[ticker]['Return'] = data_stocks_interval_roll_out[ticker]['Close'].pct_change()

    
    # Restructure the data for further analysis
    new_data_stocks_interval_roll_out = create_new_data_structure(data_stocks_interval_roll_out)

    # Calculate the covariance matrix
    covariance_matrix_roll_out = calculate_covariance_matrix(new_data_stocks_interval_roll_out)

    # Compute initial weights (w0) based on stock market capitalization
    total_market_caps_roll_out = market_caps_out['Market Cap'].sum()
    w0_roll_out = {azienda: market_caps_out.loc[azienda, 'Market Cap'] / total_market_caps_roll_out for azienda in market_caps_out.index}
    
    # --- Calculate portfolio returns ---
    results_model_1_roll_out[interval_out], results_model_2_roll_out[interval_out], results_model_3_roll_out[interval_out], index_mean_returns_roll_out, index_return_roll_out = calculate_portfolio_return(results_model_1_roll_copy[interval], results_model_2_roll_copy[interval], results_model_3_roll_copy[interval], new_data_stocks_interval_roll_out, w0_roll_out)
    
    # --- Calculate portfolio variances ---
    results_model_1_roll_out[interval_out], results_model_2_roll_out[interval_out], results_model_3_roll_out[interval_out], index_variance_roll_out = calculate_portfolio_variance(covariance_matrix_roll_out, w0_roll_out, results_model_1_roll_copy[interval], results_model_2_roll_copy[interval], results_model_3_roll_copy[interval], q_values_roll)
        
    # --- Calculate portfolio Sharpe Ratios ---
    results_model_1_roll_out[interval_out], results_model_2_roll_out[interval_out], results_model_3_roll_out[interval_out], SR_index_roll_out = calculate_sharpe_ratios(results_model_1_roll_copy[interval], results_model_2_roll_copy[interval], results_model_3_roll_copy[interval], index_return_roll_out, index_variance_roll_out, q_values_roll)
    
    row = {"Interval": interval_out, "index_return": index_return_roll_out, "index_variance": index_variance_roll_out,"SR_index": SR_index_roll_out}
    data_list_out.append(row)  # Aggiungiamo il dizionario alla lista
    
    i+=1
    interval=intervals[i]
    
    # Calculate Tracking Error
    track_error_1, track_error_2, track_error_3 = calculate_tracking_error_roll_out(q_values_roll, interval_out, covariance_matrix_roll_out, w0_roll_out, results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out)
        
    tracking_error_dict_1[interval_out]=track_error_1
    tracking_error_dict_2[interval_out]=track_error_2
    tracking_error_dict_3[interval_out]=track_error_3

# Convert the list into a DataFrame
index_return_var_out = pd.DataFrame(data_list_out)



# ------ COMPUTATIONAL RESULTS OF MODELS DYNAMIC TESTING OUT-OF-SAMPLE ------

# Plot portfolio returns
plot_portfolio_return_rolling_windows(intervals_out, index_return_var_out, q_values_roll, results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out, save_figures, results_path)
  
# Plot portfolio variances
plot_portfolio_variance_rolling_windows(intervals_out, index_return_var_out, q_values_roll, results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out, save_figures, results_path)

# Plot portfolio Sharpe Ratios
plot_portfolio_sharpe_ratios_rolling_windows(intervals_out, index_return_var_out, q_values_roll, results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out, save_figures, results_path)

# Tracking Ratios dynamic test out-of-samples 
tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3 = calculate_tracking_ratio_roll_out(q_values_roll, intervals, market_caps_dict, results_model_1_roll, results_model_1_roll_out, results_model_2_roll, results_model_2_roll_out, results_model_3_roll, results_model_3_roll_out)
plot_tracking_ratio_roll_out(q_values_roll, intervals_out, tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3, save_figures, results_path)

# Tracking Errors dynamic test out-of-samples
plot_tracking_error_roll_out(q_values_roll, intervals_out, tracking_error_dict_1, tracking_error_dict_2, tracking_error_dict_3, save_figures, results_path)
