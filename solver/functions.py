import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
from .models_factor import basic_tracking_mosek, robust_tracking_mosek, sqrt_matrix
from .models_mixture import model_EIT1, model_mixture_DEIT1
import pickle
from itertools import islice
import os

def import_data(data_path):
    """
    Function to import data
    """
    
    # Retrieve the CURRENT companies that make up the S&P 500 index from the file sp500_companies.csv
    sp500_companies = pd.read_csv(os.path.join(data_path, 'sp500_companies.csv'))

    # Select only the company symbols
    tickers = sp500_companies['Symbol'].tolist()

    # Retrieve the historical data for each component (start_date = "2019-01-01" - end_date = "2024-12-31") by reading from the file data_stocks.pkl (pickle format)
    try:
        with open(os.path.join(data_path, 'data_stocks_filtered.pkl'), 'rb') as file:
            data_stocks = pickle.load(file)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("The pickle file was not found.")
    except pickle.UnpicklingError:
        print("Error during pickle file loading.")
    
    # Import the historical market caps from the file market_caps_df_2020
    market_caps_df_2020 = pd.read_csv(os.path.join(data_path, 'market_caps_df_2020.csv'))
    market_caps_df_2020 = market_caps_df_2020.set_index('Index')
    
    # Import the historical market caps from the file market_caps_df_2021
    market_caps_df_2021 = pd.read_csv(os.path.join(data_path, 'market_caps_df_2021.csv'))
    market_caps_df_2021 = market_caps_df_2021.set_index('Index')
    
    # Import market caps for each interval
    with open(os.path.join(data_path, 'market_caps_dict.pkl'), 'rb') as f:
        market_caps_dict = pickle.load(f)

    return sp500_companies, tickers, data_stocks, market_caps_df_2020, market_caps_df_2021, market_caps_dict



def data_preprocessing(sp500_companies, tickers, data_stocks):
    """
    Function to preprocess input data
    """
    
    # --- PREPROCESSING HISTORICAL STOCKS DATA ---

    # In data_stocks, select the earliest date for each stock since we want to focus only on 2020 data.
    # Remove all stocks with an earliest date > 31/12/2019, so those that entered the index from 2020 onwards.

    # Dictionary to store the earliest date for each stock
    oldest_dates = {}

    # Loop over each stock and its associated DataFrame
    for title, df in data_stocks.items():
        # Check if the DataFrame is not empty
        if not df.empty:
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']) 
            # Ensure the first column is in datetime format
            df['Date'] = pd.to_datetime(df['Date'])

            # Find the earliest date
            oldest_date = df['Date'].min()
            
            # Add to the dictionary
            oldest_dates[title] = oldest_date
        else:
            # Handle the case where the DataFrame is empty
            oldest_dates[title] = None

    # Reference date
    reference_date = pd.Timestamp("2019-12-31 00:00:00-00:00")

    # Filter stocks with earliest date > 31 December 2019
    filtered_titles = [
        title for title, date in oldest_dates.items() 
        if pd.to_datetime(date) > reference_date
    ]

    # Remove the stocks from the data_stocks dictionary
    data_stocks = {key: value for key, value in data_stocks.items() if key not in filtered_titles}

    # Remove the stocks from the DataFrame
    sp500_companies = sp500_companies[~sp500_companies["Symbol"].isin(filtered_titles)]

    # Remove the stocks from the tickers list
    tickers = [ticker for ticker in tickers if ticker not in filtered_titles]

    return sp500_companies, tickers, data_stocks



def filter_by_date_range(data_stocks, start_date, end_date):
    """
    Function to filter data based on a date range
    """
    filtered_data = {}
    for ticker, data in data_stocks.items():
        filtered_data[ticker] = data.loc[start_date:end_date]  # Filter by dates
    return filtered_data



def create_new_data_structure(data_stocks):
    """
    Function to create a structure that contains the following information for each time period:
        - Date: Reference date
        - Ticker: Stock symbol
        - Return: Daily return
        - Mean Return: The average return of the stock
    """
    new_structure = {}
    for ticker, df in data_stocks.items():
        mean_return = df['Return'].mean()
        new_df = df[['Return']].copy()
        new_df['Mean Return'] = mean_return
        new_df.reset_index(inplace=True)
        new_structure[ticker] = new_df
    return new_structure



def calculate_average_market_caps(data_stocks_interval_1):
    """
    Function to calculate the historical market capitalization for each stock by multiplying 
    the historical prices by the number of shares outstanding.

    Input:
        data_stocks_interval_1 (dict): A dictionary with stock symbols as keys and DataFrames (with closing prices) as values.

    Output:
        dict: A dictionary with stock symbols as keys and the average market capitalizations as values.
    """
    historical_market_caps_df = {}

    for ticker, df in data_stocks_interval_1.items():
        try:
            # Retrieve the number of shares outstanding (fetch once per ticker)
            stock = yf.Ticker(ticker)
            shares_outstanding = stock.info.get('sharesOutstanding')

            if shares_outstanding:
                # Create a copy to avoid modifying the original DataFrame
                df_copy = df.copy()
                
                # Calculate the market capitalization for each day
                df_copy['Market Cap'] = df_copy['Close'] * shares_outstanding

                # Calculate the average market capitalization
                average_market_cap = df_copy['Market Cap'].mean()

                # Save the result in the dictionary
                historical_market_caps_df[ticker] = average_market_cap
            else:
                print(f" Warning: Unable to retrieve 'sharesOutstanding' for {ticker}. Skipping...")

        except Exception as e:
            print(f" Error processing {ticker}: {e}")
    
    return historical_market_caps_df



def calculate_covariance_matrix(data_stocks_interval):
    """
    Calculate the covariance matrix between the returns of all companies in the dataset.

    Inputs:
        data_stocks_interval (dict): A dictionary with stock symbols as keys and DataFrames (with closing prices) as values.

    Process:
        - For each pair of tickers, compute the covariance between their returns
          using the deviations from their mean returns.
        - Store the results in a covariance matrix, where rows and columns correspond to tickers.
        - Add a small regularization term to the diagonal for numerical stability.

    Output:
        pd.DataFrame: Covariance matrix with tickers as both row and column labels.
    """
    tickers = list(data_stocks_interval.keys())
    n_tickers = len(tickers)
    covariance_matrix = np.zeros((n_tickers, n_tickers))
    
    for i in range(n_tickers):
        for j in range(n_tickers):
            ticker_i = tickers[i]
            ticker_j = tickers[j]
            df_i = data_stocks_interval[ticker_i]
            df_j = data_stocks_interval[ticker_j]
            diff_i = df_i['Return'] - df_i['Mean Return']
            diff_j = df_j['Return'] - df_j['Mean Return']
            covariance = (diff_i * diff_j).sum() / (len(diff_i) - 1)
            covariance_matrix[i, j] = covariance

    # Regularization for numerical stability
    lambda_reg = 1e-6
    covariance_matrix += lambda_reg * np.eye(covariance_matrix.shape[0])

    return pd.DataFrame(covariance_matrix, index=tickers, columns=tickers)


def five_factor_FF(start_date, end_date, data_path):
    """
    Load the Fama-French 5-factor daily dataset and return the factor time series
    for a specified date range.

    Inputs:
        start_date (str or datetime): Start date for the filter.
        end_date (str or datetime): End date for the filter.
        data_path (str): Path to the folder containing the CSV file.

    Outputs:
        smb (np.array): Small minus Big factor series.
        hml (np.array): High minus Low factor series.
        RF (np.array): Risk-free rate series.
        mkt_RF (np.array): Market minus Risk-Free factor series.
        RMW (np.array): Robust minus Weak profitability factor series.
        CMA (np.array): Conservative minus Aggressive investment factor series.
    """

    # File name and full path
    file_name = "F-F_Research_Data_5_Factors_2x3_daily.CSV"
    file_path = os.path.join(data_path, file_name)
    
    # Load CSV, skipping the first 4 rows, and drop missing values
    df_2 = pd.read_csv(file_path, skiprows=4)
    df_2 = df_2.dropna()
    
    # Rename columns to standard factor names
    df_2.rename(columns={df_2.columns[0]: "Date"}, inplace=True)
    df_2.columns = ["Date","Mkt-RF","SMB","HML","RMW","CMA","RF"]
    
    # Convert 'Date' to datetime and localize to UTC; convert numeric columns
    df_2["Date"] = pd.to_datetime(df_2["Date"], format="%Y%m%d").dt.tz_localize("UTC")
    df_2.iloc[:, 1:] = df_2.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")

    # Convert input start and end dates to UTC timestamps
    start_date = pd.to_datetime(start_date).tz_localize("UTC")
    end_date = pd.to_datetime(end_date).tz_localize("UTC")

    # Filter the dataset to the specified date range
    df_2_filtered = df_2[(df_2["Date"] >= start_date) & (df_2["Date"] <= end_date)]

    # Extract factor series as numpy arrays
    smb = df_2_filtered['SMB'].to_numpy()
    hml = df_2_filtered['HML'].to_numpy()
    RF = df_2_filtered['RF'].to_numpy()
    mkt_RF = df_2_filtered['Mkt-RF'].to_numpy()
    RMW = df_2_filtered['RMW'].to_numpy()
    CMA = df_2_filtered['CMA'].to_numpy()
    
    return smb, hml, RF, mkt_RF, RMW, CMA


def estimates_parameters_general(factors_matrix, new_data_stocks_interval_1, omega=0.95):
    """
    General parameter estimation function using linear regression.

    Parameters:
    - factors_matrix: (p x m) matrix of factor data (e.g., mkt_RF, RF, SMB, HML, ...)
    - new_data_stocks_interval_1: dict containing stock return series
    - omega: confidence level (default = 0.95)

    Returns:
    - mu_0: prior expected return for each stock (n x 1)
    - V0: factor loadings matrix (m x n)
    - rho: uncertainty bound on alpha (n x 1)
    - gamma_up: upper bound for each factor loading (n x 1)
    - gamma_down: lower bound for each factor loading (n x 1)
    - G: covariance matrix used in robust optimization (m x m)
    - F: estimated factor covariance matrix (m x m)
    - d_up: max residual variance across all stocks (scalar)
    """

    eps = 1e-6
    B = factors_matrix[1:].T  # transpose after skipping first row (due to NaN)
    m, p = B.shape  # m = number of factors, p = number of periods
    A = np.column_stack((np.ones((p, 1)), B.T))  # add intercept term

    # Build matrix Q to extract factor loadings from x_hat
    QT = np.zeros((m + 1, m))
    for i in range(m):
        QT[:, i] = np.eye(m + 1)[i + 1]
    Q = QT.T

    # Initialize result containers
    y, x_hat, mu_0, s2 = {}, {}, {}, {}
    d_up, gamma_up, gamma_down, V0, rho = {}, {}, {}, {}, {}

    for key in new_data_stocks_interval_1:
        y[key] = new_data_stocks_interval_1[key]["Return"].iloc[1:].values
        Z = np.linalg.solve(A.T @ A + eps * np.eye(A.shape[1]), np.eye(A.shape[1]))
        x_hat[key] = Z @ A.T @ y[key]
        s2[key] = np.sum((y[key] - (A @ x_hat[key]))**2) / (p - m - 1)
        mu_0[key] = x_hat[key][0]
        V0[key] = x_hat[key][1:]
        
        c_1 = stats.f.ppf(omega, 1, p - m - 1)
        c_m = stats.f.ppf(omega, m, p - m - 1)
        
        K = Z[1, 1] * c_1 * s2[key]
        gamma_up[key] = np.sqrt(K)
        gamma_down[key] = -np.sqrt(K)
        rho[key] = np.sqrt(m * c_m * s2[key])
        d_up[key] = np.max(s2[key]) #change

    # Compute G and F matrices
    G = np.linalg.solve(Q @ Z @ QT + eps * np.eye(Q.shape[0]), np.eye(Q.shape[0]))
    F = (1 / (p - 1)) * G

    # Convert results from dictionaries to arrays
    mu_0 = np.array(list(mu_0.values())).reshape(-1, 1)
    rho = np.array(list(rho.values())).reshape(-1, 1)
    gamma_down = np.array(list(gamma_down.values())).reshape(-1, 1)
    gamma_up = np.array(list(gamma_up.values())).reshape(-1, 1)
    V0T = np.array(list(V0.values()))
    V0 = V0T.T
    s2 = np.array(list(s2.values())).reshape(-1, 1)
    d_up = np.array(list(d_up.values())).reshape(-1, 1) 

    return mu_0, V0, rho, gamma_up, gamma_down, G, F, d_up

def save_model_results_mosek(q_values, mu, Cov_sqrt, w0_1, sigma, TE, market_caps_df_1, mu_0, gamma_down, F, V0, rho, G, d_up, 
                             mu_0_5, gamma_down_5, F_5, V0_5, rho_5, G_5, d_up_5):
    """
    Function to save the results of the 4 models as q values change in the dictionaries results_model_1, ..., results_model_4.

    Input:
    - q_values: list of q values
    - rho: correlation matrix
    - market_caps_df_1: DataFrame with market capitalizations
    - w0_1: initial weights vector
    - sector_counts: counts of companies per sector
    - sector_correlation_matrices: dictionary with correlation matrices per sector
    - sector_companies_dict: dictionary with DataFrames of companies per sector

    Output:
    - results_model_1, results_model_2, results_model_3, results_model_4: dictionaries with the model results
    """

    # Initialization of the dictionaries for the results
    results_1 = []
    results_2 = []
    results_3 = []
    mosek_result_1 = {}
    mosek_result_2 = {}
    mosek_result_3 = {}
    prev_solution=None

    # Iterates over all q values
    for q in q_values:
        # Calculates the results for each model
        obj_val_1, weights_1, selected_assets_1, norm_diff_1, time_1 = basic_tracking_mosek(mu, Cov_sqrt, market_caps_df_1, w0_1, sigma, TE, q)
        obj_val_2, weights_2, selected_assets_2, norm_diff_2, time_2, prev_solution = robust_tracking_mosek(mu_0, gamma_down, F, V0, rho, G, d_up, market_caps_df_1, w0_1, sigma, TE, q, prev_solution)
        obj_val_3, weights_3, selected_assets_3, norm_diff_3, time_3, prev_solution = robust_tracking_mosek(mu_0_5, gamma_down_5, F_5, V0_5, rho_5, G_5, d_up_5, market_caps_df_1, w0_1, sigma, TE, q, prev_solution)
        # Saves the results in their respective dictionaries
        mosek_result_1[q] = [obj_val_1, weights_1, selected_assets_1, norm_diff_1]
        mosek_result_2[q] = [obj_val_2, weights_2, selected_assets_2, norm_diff_2]
        mosek_result_3[q] = [obj_val_3, weights_3, selected_assets_3, norm_diff_3]
        results_1.append({"Dimensione": q, "Tempo Totale (s)": time_1})
        results_2.append({"Dimensione": q, "Tempo Totale (s)": time_2})
        results_3.append({"Dimensione": q, "Tempo Totale (s)": time_3})
        
    df_1 = pd.DataFrame(results_1)
    df_2 = pd.DataFrame(results_2)
    df_3 = pd.DataFrame(results_3)

    return mosek_result_1, mosek_result_2, mosek_result_3, df_1, df_2, df_3    


def calculate_portfolio_return(results_model_1, results_model_2, results_model_3, new_data_stocks_interval_1, w0_1):
     """
     Function that modifies the dictionaries results_model_1, results_model_2 and results_model_3
     by updating their DataFrames with a new column "Mean Return" (For each DataFrame associated with the keys in the 
     dictionaries results_model_1, results_model_2, and results_model_3, the "Mean Return" column is added 
     using the mapping from the dictionary index_mean_returns_1).
     The summation of the product between "Weight" and "Mean Return" is calculated. This result is added as an additional element 
     in the list associated with the key.
     
     Input:
         - results_model_1, results_model_2, results_model_3: dictionaries containing the DataFrames 
           with the results for each model.
         - new_data_stocks_interval_1: dictionary containing the data for average returns for each stock.
     
     Output:
         - results_model_1, results_model_2, results_model_3: dictionaries containing the DataFrames 
           with the results for each model.
         - index_mean_returns_1 (dictionary): Contains the mean returns for each stock.
     """
     
     # MODEL 1 - INTERVAL 1

     # Dictionary to associate each stock with its mean returns
     index_mean_returns_1 = {}

     # Iterate over the results_model_1 dictionary
     for q, result in results_model_1.items():
         df_result = result[1]   #give us the weights for each q
         
         for title, stock_data in new_data_stocks_interval_1.items():
             if 'Mean Return' in stock_data.columns:
                 stock_data = stock_data.dropna(subset=['Mean Return'])
                 if not stock_data.empty:
                     mean_return = stock_data['Mean Return'].iloc[0]
                     index_mean_returns_1[title] = mean_return
                 else:
                     print(f"Stock: {title}, the 'Mean Return' column contains only NaN or the DataFrame is empty!")
             else:
                 print(f"Stock: {title}, the 'Mean Return' column does not exist!")

         df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)  #add new column
         results_model_1[q][1] = df_result  #now here we have weights and mean returns

     for q, result in results_model_1.items():
         try:
             df_result = result[1]
             if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                 sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                 results_model_1[q].append(sum_product * 100)   #add value: weight'*mean return
             else:
                 print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
         except Exception as e:
             print(f"Error in key {q}: {e}")

     # MODEL 2 - INTERVAL 1
     for q, result in results_model_2.items():
         df_result = result[1]
         df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)
         results_model_2[q][1] = df_result

     for q, result in results_model_2.items():
         try:
             df_result = result[1]
             if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                 sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                 results_model_2[q].append(sum_product * 100)
             else:
                 print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
         except Exception as e:
             print(f"Error in key {q}: {e}")

     # MODEL 3 - INTERVAL 1
     for q, result in results_model_3.items():
         df_result = result[1]
         df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)
         results_model_3[q][1] = df_result

     for q, result in results_model_3.items():
         try:
             df_result = result[1]
             if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
                 sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
                 results_model_3[q].append(sum_product * 100)
             else:
                 print(f"Key {q}: The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
         except Exception as e:
             print(f"Error in key {q}: {e}")
     
     # Calculate the 2020 annual return for the S&P 500 index - Interval 1  (it's in percentage)
     index_return_1 = sum(index_mean_returns_1[title] * w0_1[title] for title in index_mean_returns_1 if title in w0_1) * 100


     return results_model_1, results_model_2, results_model_3, index_mean_returns_1, index_return_1



def calculate_portfolio_variance(covariance_matrix_1, w0_1, results_model_1, results_model_2, results_model_3, q_values):
     """
     Function to calculate the portfolio variances.

     Input:
     - covariance_matrix_1 (pd.DataFrame): Full covariance matrix.
     - w0_1 (dict): Dictionary with stock symbols and index weights.
     - results_model_1, results_model_2, results_model_3: model results.
     - q_values (list): Portfolio sizes.

     Output:
         - results_model_1, results_model_2, results_model_3: model results.
         - index_variance: variances of the S&P 500 index.
     """
     
     results_models = [results_model_1, results_model_2, results_model_3]
     
     # Calculate variance for the index
     tickers_index = list(w0_1.keys())
     weights_index = np.array(list(w0_1.values()))  #we take only the values of the weights
     cov_submatrix_index = covariance_matrix_1.loc[tickers_index, tickers_index]   #pull out the submatrix
     index_variance = np.dot(weights_index, np.dot(cov_submatrix_index, weights_index))  #w'*Sigma*w

     # Update models 1 and 2 with portfolio variance
     for model in results_models:  # Models 1, 2, 3
         for q, result in model.items():
             df_results = result[1]
             tickers_ptf = df_results['Stock'].tolist()   #list of tickers
             weights_ptf = df_results['Weight'].values
             cov_submatrix_ptf = covariance_matrix_1.loc[tickers_ptf, tickers_ptf]
             ptf_var = np.dot(weights_ptf, np.dot(cov_submatrix_ptf, weights_ptf))
             result.append(ptf_var)
             #print(f"Model 1/2, q={q}, portfolio variance: {ptf_var}")

     return results_model_1, results_model_2, results_model_3, index_variance



def calculate_sharpe_ratios(results_model_1, results_model_2, results_model_3, index_return_1, index_variance_1, q_values):
     """
     Function to calculate the Sharpe Ratios of the portfolios.

     Input:
     - results_model_1, results_model_2, results_model_3: model results.
     - index_return_1: return of the S&P 500 index.
     - index_variance_1: variance of the S&P 500 index.
     - q_values (list): Portfolio sizes.

     Output:
         - results_model_1, results_model_2, results_model_3: model results.
         - SR_index: Sharpe ratio of the S&P 500 index.
     """
     
     results_models = [results_model_1, results_model_2, results_model_3]

     # Average closing rate for the 3-month Treasury throughout 2020(it's considered a good approximation of the annual risk-free rate)
     average_treasury_rate_2020 = 0.0033613095265288377

     # Calculate the daily risk-free rate for 2020(effective risk-free rate, 252 are the typical number of trading days in a year in the financial markets)
     risk_free_daily_2020 = (1 + average_treasury_rate_2020) ** (1 / 252) - 1

     # Calculate the annualized Sharpe ratio for 2020 (remember that the returns are dailys)
     SR_index = (index_return_1 / 100 - risk_free_daily_2020) / np.sqrt(index_variance_1) * np.sqrt(252)  

     # Calculate and store the Sharpe ratio for each model
     for model in results_models:
         for q, result in model.items():
             SR_ptf = (result[4] / 100 - risk_free_daily_2020) / np.sqrt(result[5]) * np.sqrt(252)   #are the positions rights?
             result.append(SR_ptf)

     return results_model_1, results_model_2, results_model_3, SR_index 


def return_comparison_in_out(q_values, results_model_1, results_model_1_out, 
                             results_model_2, results_model_2_out, 
                             results_model_3, results_model_3_out):
    """
    Function to perform an analytical comparison between the returns of the 4 models with in-sample and out-of-sample data.
    
    Input:
    - q_values: List of values for the 'q' column.
    - results_model_1, results_model_2, results_model_3: Dictionaries with the results of the models for in-sample data.
    - results_model_1_out, results_model_2_out, results_model_3_out: Dictionaries with the results of the models for out-of-sample data.
    
    Output:
    - return_models_in_out: DataFrame with the results of the comparison between the models.
    """

    # Initialization of the empty DataFrame
    return_models_in_out = pd.DataFrame({
        'q': q_values,
        'model_1_in_samples': [None] * len(q_values),
        'model_1_out_samples': [None] * len(q_values),
        'diff_1': [None] * len(q_values),
        'model_2_in_samples': [None] * len(q_values),
        'model_2_out_samples': [None] * len(q_values),
        'diff_2': [None] * len(q_values),
        'model_3_in_samples': [None] * len(q_values),
        'model_3_out_samples': [None] * len(q_values),
        'diff_3': [None] * len(q_values)
    })

    # Loop to assign values to the columns for each q value
    for q in q_values:
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_1_in_samples'] = results_model_1[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_1_out_samples'] = results_model_1_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_1'] = results_model_1_out[q][4] - results_model_1[q][4]

        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_2_in_samples'] = results_model_2[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_2_out_samples'] = results_model_2_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_2'] = results_model_2_out[q][4] - results_model_2[q][4]

        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_3_in_samples'] = results_model_3[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'model_3_out_samples'] = results_model_3_out[q][4]
        return_models_in_out.loc[return_models_in_out['q'] == q, 'diff_3'] = results_model_3_out[q][4] - results_model_3[q][4]

    return return_models_in_out



def variance_comparison_in_out(q_values, results_model_1, results_model_1_out, 
                             results_model_2, results_model_2_out, 
                             results_model_3, results_model_3_out):
    """
    Function to perform an analytical comparison between the variances of the 3 models with in-sample and out-of-sample data.
    
    Input:
    - q_values: List of values for the 'q' column.
    - results_model_1, results_model_2, results_model_3: Dictionaries with the results of the models for in-sample data.
    - results_model_1_out, results_model_2_out, results_model_3_out: Dictionaries with the results of the models for out-of-sample data.
    
    Output:
    - variance_models_in_out: DataFrame with the results of the comparison between the models.
    """

    # Initialization of the empty DataFrame
    variance_models_in_out = pd.DataFrame({
        'q': q_values,
        'model_1_in_samples': [None] * len(q_values),
        'model_1_out_samples': [None] * len(q_values),
        'diff_1': [None] * len(q_values),
        'model_2_in_samples': [None] * len(q_values),
        'model_2_out_samples': [None] * len(q_values),
        'diff_2': [None] * len(q_values),
        'model_3_in_samples': [None] * len(q_values),
        'model_3_out_samples': [None] * len(q_values),
        'diff_3': [None] * len(q_values)
    })

    # Loop to assign values to the columns for each q value
    for q in q_values:
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_1_in_samples'] = results_model_1[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_1_out_samples'] = results_model_1_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_1'] = results_model_1_out[q][5] - results_model_1[q][5]

        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_2_in_samples'] = results_model_2[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_2_out_samples'] = results_model_2_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_2'] = results_model_2_out[q][5] - results_model_2[q][5]

        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_3_in_samples'] = results_model_3[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'model_3_out_samples'] = results_model_3_out[q][5]
        variance_models_in_out.loc[variance_models_in_out['q'] == q, 'diff_3'] = results_model_3_out[q][5] - results_model_3[q][5]

    return variance_models_in_out



def sharpe_ratios_comparison_in_out(q_values, results_model_1, results_model_1_out, 
                             results_model_2, results_model_2_out, 
                             results_model_3, results_model_3_out):
    """
    Function to perform an analytical comparison between the Sharpe ratios of the 3 models with in-sample and out-of-sample data.
    
    Input:
    - q_values: List of values for the 'q' column.
    - results_model_1, results_model_2, results_model_3: Dictionaries with the results of the models for in-sample data.
    - results_model_1_out, results_model_2_out, results_model_3_out: Dictionaries with the results of the models for out-of-sample data.
    
    Output:
    - sharpe_ratios_models_in_out: DataFrame with the results of the comparison between the models.
    """

    # Initialization of the empty DataFrame
    sharpe_ratios_models_in_out = pd.DataFrame({
        'q': q_values,
        'model_1_in_samples': [None] * len(q_values),
        'model_1_out_samples': [None] * len(q_values),
        'diff_1': [None] * len(q_values),
        'model_2_in_samples': [None] * len(q_values),
        'model_2_out_samples': [None] * len(q_values),
        'diff_2': [None] * len(q_values),
        'model_3_in_samples': [None] * len(q_values),
        'model_3_out_samples': [None] * len(q_values),
        'diff_3': [None] * len(q_values)
    })

    # Loop to assign values to the columns for each q value
    for q in q_values:
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_1_in_samples'] = results_model_1[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_1_out_samples'] = results_model_1_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_1'] = results_model_1_out[q][6] - results_model_1[q][6]

        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_2_in_samples'] = results_model_2[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_2_out_samples'] = results_model_2_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_2'] = results_model_2_out[q][6] - results_model_2[q][6]

        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_3_in_samples'] = results_model_3[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'model_3_out_samples'] = results_model_3_out[q][6]
        sharpe_ratios_models_in_out.loc[sharpe_ratios_models_in_out['q'] == q, 'diff_3'] = results_model_3_out[q][6] - results_model_3[q][6]

    return sharpe_ratios_models_in_out



def calculate_tracking_ratio(results_model_1, results_model_1_out, results_model_2, results_model_2_out, results_model_3, results_model_3_out, market_caps_df_1, market_caps_df_2, total_market_caps_1, total_market_caps_2, q_values):
    """
    Calculate tracking ratio in-sample and out-of-sample for different models.
    The tracking ratio (R0t) is calculated as the ratio between the performance of the reference index (S&P 500)
    and the performance of the tracking portfolio over a given period.
    Formula: R0t = ( ΣVit / ΣVi0 ) / ( ΣwjVjt / ΣwjVj0 )
    where:
    - ΣVit: sum of market values of all assets in the reference index (S&P 500) at time t.
    - ΣVi0: sum of market values of all assets in the reference index at the initial time (time 0).
    - ΣwjVjt: sum of market values of assets in the tracking portfolio at time t, weighted by their proportion.
    - ΣwjVj0: sum of market values of assets in the tracking portfolio at time 0, weighted by their initial proportion.
    
    Input:
        - results_model_1, results_model_2, results_model_3: Dictionaries with the results of the models for in-sample data.
        - results_model_1_out, results_model_2_out, results_model_3_out: Dictionaries with the results of the models for out-of-sample data.
        - market_caps_df_1: DataFrame with market capitalizations of assets of interval 1 (in samples data)
        - market_caps_df_2: DataFrame with market capitalizations of assets of interval 2 (out of samples data)
        - total_market_caps_1: Total market capitalizations of assets of interval 1 (in samples data)
        - total_market_caps_2: Total market capitalizations of assets of interval 2 (out of samples data)
        - q_values: List of values for the 'q' column.
    
    Output:
        - tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3: Tracking ratios values for models for different values of q
    """
    tracking_ratio_model_1=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_1[q][1].merge(market_caps_df_1, left_on='Stock', right_index=True)
        df_merged_out = results_model_1_out[q][1].merge(market_caps_df_2, left_on='Stock', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_1.loc[tracking_ratio_model_1['q'] == q, 'tracking_ratio'] = tracking_ratio

    tracking_ratio_model_2=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_2[q][1].merge(market_caps_df_1, left_on='Stock', right_index=True)
        df_merged_out = results_model_2_out[q][1].merge(market_caps_df_2, left_on='Stock', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_2.loc[tracking_ratio_model_2['q'] == q, 'tracking_ratio'] = tracking_ratio

    tracking_ratio_model_3=pd.DataFrame({'q': q_values,'tracking_ratio': [None] * len(q_values)})

    for q in q_values:
        df_merged_1 = results_model_3[q][1].merge(market_caps_df_1, left_on='Stock', right_index=True)
        df_merged_out = results_model_3_out[q][1].merge(market_caps_df_2, left_on='Stock', right_index=True)
        # Calculate the sum of the product Weight * Market Cap
        somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
        somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
        
        tracking_ratio=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
        tracking_ratio_model_3.loc[tracking_ratio_model_3['q'] == q, 'tracking_ratio'] = tracking_ratio



    return tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3



def calculate_tracking_error(q_values, w0_2021, covariance_matrix_2021, results_model_1_out, results_model_2_out, results_model_3_out):
    """
    Calculate tracking error for static test with out-of-samples data
    
    Input:
        -q_values (list): List of q values
        -w0_2021 (dict): Dictionary of index weight of 2021
        -covariance_matrix_2021: Covariance matrix of 2021
        -results_model_1_out, results_model_2_out, results_model_3_out: Results of models for static test of out-of-samples 2021 data
    
    Output:
        -tracking_error_model_1, tracking_error_model_2, tracking_error_model_3: Tracking errors values for models for different values of q
    """
    
    tracking_error_model_1=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    tracking_error_model_2=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    tracking_error_model_3=pd.DataFrame({'q': q_values,'tracking_error': [None] * len(q_values)})
    
    for q in q_values:
        
        # MODEL 1
        df_index = pd.DataFrame(list(w0_2021.items()), columns=["Stock", "index_weight"])
        
        df_merged_1 = results_model_1_out[q][1].merge(df_index, on='Stock', how='left')
        df_merged_1['diff']=df_merged_1['Weight']-df_merged_1['index_weight']
        
        tickers_ptf = df_merged_1['Stock'].tolist()
        diff_ptf = df_merged_1['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_1 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_1.loc[tracking_error_model_1['q'] == q, 'tracking_error'] = tracking_error_1
        
        # MODEL 2
        df_merged_2 = results_model_2_out[q][1].merge(df_index, on='Stock', how='left')
        df_merged_2['diff']=df_merged_2['Weight']-df_merged_2['index_weight']
        
        tickers_ptf = df_merged_2['Stock'].tolist()
        diff_ptf = df_merged_2['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_2 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_2.loc[tracking_error_model_2['q'] == q, 'tracking_error'] = tracking_error_2
        
        # MODEL 3        
        df_merged_3 = results_model_3_out[q][1].merge(df_index, on='Stock', how='left')
        df_merged_3['diff']=df_merged_3['Weight']-df_merged_3['index_weight']
        
        tickers_ptf = df_merged_3['Stock'].tolist()
        diff_ptf = df_merged_3['diff'].values
        cov_submatrix_ptf = covariance_matrix_2021.loc[tickers_ptf, tickers_ptf]
        tracking_error_3 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        tracking_error_model_3.loc[tracking_error_model_3['q'] == q, 'tracking_error'] = tracking_error_3
        

    return tracking_error_model_1, tracking_error_model_2, tracking_error_model_3






def perform_rolling_analysis(market_caps_dict, data_stocks, sp500_companies, q_values_roll, data_path):
    """
    Performs a test on data out-of-samples with a rolling windows analysis on different time intervals by iterating through predefined market cap intervals.
    Computes returns, restructures data, calculates covariance and correlation matrices, and runs four different models.
    
    Input:
        - market_caps_dict (dict): Dictionary with time intervals as keys and market cap DataFrames as values.
        - data_stocks (dict): Dictionary with stock symbols as keys and their historical data as values.
        - sp500_companies (DataFrame): DataFrame containing sector information for S&P 500 companies.
        - q_values_roll (list): List of portfolio sizes to evaluate.
    
    Output:
        - dict: Four dictionaries containing the results for each model across all intervals and portfolio sizes.
    """
    results_model_1_roll = {}
    results_model_2_roll = {}
    results_model_3_roll = {}
    prev_solution_3 = None
    prev_solution_5 = None
    #ten_interval_dict = dict(list(market_caps_dict.items())[9:10])

    # Iterate through each time interval
    for interval, market_caps in market_caps_dict.items():
    #for interval, market_caps in ten_interval_dict.items():    
        results_model_1_q = {}
        results_model_2_q = {}
        results_model_3_q = {}
        
        # Create a new filtered dictionary for the interval
        data_stocks_interval_roll = filter_by_date_range(data_stocks, *interval)

        # Compute percentage returns for each stock in the filtered dictionary
        for ticker, df in data_stocks_interval_roll.items():
            df = df.copy()
            df['Return'] = df['Close'].pct_change()
            data_stocks_interval_roll[ticker] = df  # salva la versione modificata

        # Restructure the data for further analysis
        new_data_stocks_interval_roll = create_new_data_structure(data_stocks_interval_roll)
        
        # data_stocks_interval_roll_out è il dict dei DataFrame per l'intervallo 9
        for t, df in data_stocks_interval_roll.items():
            if df['Close'].isna().any():
                print("NaN in Close per", t)
            if df['Return'].isna().all():
                print("Tutti NaN nei returns per", t)


        # Calculate the covariance matrix
        covariance_matrix_roll = calculate_covariance_matrix(new_data_stocks_interval_roll)

        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll = market_caps['Market Cap'].sum()
        w0_roll = {azienda: market_caps.loc[azienda, 'Market Cap'] / total_market_caps_roll for azienda in market_caps.index}
        
        # Parameters for models
        start_date_roll, end_date_roll = interval
        smb, hml, RF, mkt_RF, RMW, CMA = five_factor_FF(start_date_roll, end_date_roll, data_path)

        # Data used for the basic model
        # Define mu
        keys_roll = list(new_data_stocks_interval_roll.keys())

        # Creiamo un vettore vuoto di lunghezza n
        n = len(new_data_stocks_interval_roll)
        mu_roll = np.zeros(n)

        # Ciclo per inserire i valori nella posizione corretta
        for i, key in enumerate(keys_roll):
            mu_roll[i] = new_data_stocks_interval_roll[key]["Mean Return"].iloc[0]

        Cov_sqrt_roll = sqrt_matrix(covariance_matrix_roll)

        #we need the maximal standard deviation for the parameters sigma and TE
        std_dev_roll = np.sqrt(np.diag(covariance_matrix_roll))
        perc = 95                                          
        p_std_roll = np.percentile(std_dev_roll, perc)

        sigma_roll = 6*p_std_roll
        TE_roll = 3*p_std_roll
        
        factors_matrix_3_roll = np.column_stack((mkt_RF, RF, smb, hml))  # (253 x 4)
        mu_0_roll, V0_roll, rho_roll, gamma_up_roll, gamma_down_roll, G_roll, F_roll, d_up_roll = estimates_parameters_general(factors_matrix_3_roll, new_data_stocks_interval_roll)

        factors_matrix_5_roll = np.column_stack((mkt_RF, RF, smb, hml, RMW, CMA))  # (253 x 6)
        mu_0_5_roll, V0_5_roll, rho_5_roll, gamma_up_5_roll, gamma_down_5_roll, G_5_roll, F_5_roll, d_up_5_roll = estimates_parameters_general(factors_matrix_5_roll, new_data_stocks_interval_roll)


        # Iterate over different portfolio sizes
        for q in q_values_roll:
            # Run the four models for varying portfolio sizes and store the results
            obj_val_1_roll, weights_1_roll, selected_assets_1_roll, norm_diff_1_roll, time_1_roll = basic_tracking_mosek(mu_roll, Cov_sqrt_roll, market_caps, w0_roll, sigma_roll, TE_roll, q)
            obj_val_2_roll, weights_2_roll, selected_assets_2_roll, norm_diff_2_roll, time_2,prev_solution_3 = robust_tracking_mosek(mu_0_roll, gamma_down_roll, F_roll, V0_roll, rho_roll, G_roll, d_up_roll, market_caps, w0_roll, sigma_roll, TE_roll, q, prev_solution_3)
            obj_val_3_roll, weights_3_roll, selected_assets_3_roll, norm_diff_3_roll, time_3, prev_solution_5 = robust_tracking_mosek(mu_0_5_roll, gamma_down_5_roll, F_5_roll, V0_5_roll, rho_5_roll, G_5_roll, d_up_5_roll, market_caps, w0_roll, sigma_roll, TE_roll, q, prev_solution_5)

            #models's results
            results_model_1_q[q] = [obj_val_1_roll, weights_1_roll, selected_assets_1_roll, norm_diff_1_roll]
            results_model_2_q[q] = [obj_val_2_roll, weights_2_roll, selected_assets_2_roll, norm_diff_2_roll]
            results_model_3_q[q] = [obj_val_3_roll, weights_3_roll, selected_assets_3_roll, norm_diff_3_roll]

        # Store the results for the current interval
        results_model_1_roll[interval] = results_model_1_q
        results_model_2_roll[interval] = results_model_2_q
        results_model_3_roll[interval] = results_model_3_q
    
    return results_model_1_roll, results_model_2_roll, results_model_3_roll



def calculate_tracking_ratio_roll_out(q_values_roll, intervals, market_caps_dict, results_model_1_roll, results_model_1_roll_out, results_model_2_roll, results_model_2_roll_out, results_model_3_roll, results_model_3_roll_out):
    """
    Calculate tracking ratios for out-of-samples dynamic test
    
    Input:
        - q_values_roll: List of q values ofr dynamic test
        - intervals: List of intervals for dynamic test
        - market_caps_dict: Dictionary opf market caps values for each interval
        - results_model_1_roll, results_model_2_roll, results_model_3_roll: Results of the models for each intervals in-samples data 
        - results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out: Results of the models for each intervals out-of-samples data
    
    Output:
        -tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3: Tracking ratios for each models in each intervals
    """
    tracking_ratio_dict_1={}
    tracking_ratio_dict_2={}
    tracking_ratio_dict_3={}

    interval=intervals[0]
    i=0

    for interval_out, market_caps_out in islice(market_caps_dict.items(), 4, None):
    
        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll = market_caps_dict[interval]['Market Cap'].sum()
    
        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll_out = market_caps_out['Market Cap'].sum()
    
    
        # Calculate tracking ratios in-sample and out-of-sample
        tracking_ratio_model_1_roll_out, tracking_ratio_model_2_roll_out, tracking_ratio_model_3_roll_out = calculate_tracking_ratio(results_model_1_roll[interval], results_model_1_roll_out[interval_out], results_model_2_roll[interval], results_model_2_roll_out[interval_out], results_model_3_roll[interval], results_model_3_roll_out[interval_out], market_caps_dict[interval], market_caps_out, total_market_caps_roll, total_market_caps_roll_out, q_values_roll)
    
        tracking_ratio_dict_1[interval_out]=tracking_ratio_model_1_roll_out
        tracking_ratio_dict_2[interval_out]=tracking_ratio_model_2_roll_out
        tracking_ratio_dict_3[interval_out]=tracking_ratio_model_3_roll_out
        
        i+=1
        interval=intervals[i]

    return tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3
    


def calculate_tracking_error_roll_out(q_values_roll, interval_out, covariance_matrix_roll_out, w0_roll_out, results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out):
    """
    Calculate tracking error for out-of-samples dynamic test
    
    Input:
        -q_values_roll: List of q values ofr dynamic test
        -interval_out: Interval for dynamic test
        -covariance_matrix_roll_out: Covariance matrix of specified interval
        -w0_roll_out: Index weight for specified interval
        -results_model_1_roll_out, results_model_2_roll_out, results_model_3_roll_out: Results of models out-of-sample
    
    Output:
        -track_error_1, track_error_2, track_error_3: Tracking error of each models for a specified time interval as various q values
    """
    
    track_error_1={}
    track_error_2={}
    track_error_3={}
    
    for q in q_values_roll:
        df_index_1_2 = pd.DataFrame(list(w0_roll_out.items()), columns=["Stock", "index_weight"])
        
        # MODEL 1
        df_merged_1 = results_model_1_roll_out[interval_out][q][1].merge(df_index_1_2, on='Stock', how='left')
        df_merged_1['diff']=df_merged_1['Weight']-df_merged_1['index_weight']
    
        tickers_ptf = df_merged_1['Stock'].tolist()
        diff_ptf = df_merged_1['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_1 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_1[q]=tracking_error_1
        
        # MODEL 2
        df_merged_2 = results_model_2_roll_out[interval_out][q][1].merge(df_index_1_2, on='Stock', how='left')
        df_merged_2['diff']=df_merged_2['Weight']-df_merged_2['index_weight']
    
        tickers_ptf = df_merged_2['Stock'].tolist()
        diff_ptf = df_merged_2['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_2 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_2[q]=tracking_error_2
        
        # MODEL 3        
        df_merged_3 = results_model_3_roll_out[interval_out][q][1].merge(df_index_1_2, on='Stock', how='left')
        df_merged_3['diff']=df_merged_3['Weight']-df_merged_3['index_weight']
    
        tickers_ptf = df_merged_3['Stock'].tolist()
        diff_ptf = df_merged_3['diff'].values
        cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
        tracking_error_3 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
        
        track_error_3[q]=tracking_error_3
      

    return track_error_1, track_error_2, track_error_3


def save_model_results_gurobi(lamb, mu, cov, market_caps_df, w0_1, new_data_stocks_interval_1):
    """
    Function to save the results of the 4 models as q values change in the dictionaries results_model_1, ..., results_model_4.

    Input:
    - q_values: list of q values
    - rho: correlation matrix
    - market_caps_df_1: DataFrame with market capitalizations
    - w0_1: initial weights vector
    - sector_counts: counts of companies per sector
    - sector_correlation_matrices: dictionary with correlation matrices per sector
    - sector_companies_dict: dictionary with DataFrames of companies per sector

    Output:
    - results_model_1, results_model_2, results_model_3, results_model_4: dictionaries with the model results
    """

    # Initialization of the dictionaries for the results
    results_model_1 = []
    results_model_2 = []
    # Calculates the results for each model
    obj_val_1, weights_1, selected_assets_1, norm_diff_1, time_1 = model_EIT1(lamb, mu, cov, market_caps_df, w0_1, new_data_stocks_interval_1)
    obj_val_2, weights_2, selected_assets_2, norm_diff_2, time_2 = model_mixture_DEIT1(lamb, mu, cov, market_caps_df, w0_1, new_data_stocks_interval_1)
    # Saves the results in their respective dictionaries
    results_model_1 = [obj_val_1, weights_1, selected_assets_1, norm_diff_1]
    results_model_2 = [obj_val_2, weights_2, selected_assets_2, norm_diff_2]

    return results_model_1, results_model_2


def calculate_portfolio_return_mixture(results_model_1, results_model_2, new_data_stocks_interval_1, w0_1):
     """
     Function that modifies the dictionaries results_model_1 and results_model_2
     by updating their DataFrames with a new column "Mean Return" (For each DataFrame associated with the keys in the 
     dictionaries results_model_1, and results_model_2, the "Mean Return" column is added 
     using the mapping from the dictionary index_mean_returns_1).
     The summation of the product between "Weight" and "Mean Return" is calculated. This result is added as an additional element 
     in the list associated with the key.
     
     Input:
         - results_model_1, results_model_2: dictionaries containing the DataFrames 
           with the results for each model.
         - new_data_stocks_interval_1: dictionary containing the data for average returns for each stock.
     
     Output:
         - results_model_1, results_model_2: dictionaries containing the DataFrames 
           with the results for each model.
         - index_mean_returns_1 (dictionary): Contains the mean returns for each stock.
     """
     
     # MODEL 1 - INTERVAL 1

     # Dictionary to associate each stock with its mean returns
     index_mean_returns_1 = {}
     
     df_result = results_model_1[1]   #give us the weights
    
     for title, stock_data in new_data_stocks_interval_1.items():
         if 'Mean Return' in stock_data.columns:
             stock_data = stock_data.dropna(subset=['Mean Return'])
             if not stock_data.empty:
                 mean_return = stock_data['Mean Return'].iloc[0]
                 index_mean_returns_1[title] = mean_return
             else:
                 print(f"Stock: {title}, the 'Mean Return' column contains only NaN or the DataFrame is empty!")
         else:
             print(f"Stock: {title}, the 'Mean Return' column does not exist!")

     df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)  # add new column
     results_model_1[1] = df_result  # now here we have weights and mean returns

     try:
         df_result = results_model_1[1]
         if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
             sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
             results_model_1.append(sum_product * 100)   # add value: weight'*mean return
         else:
             print("The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
     except Exception as e:
         print(f"Error: {e}")

     # MODEL 2 - INTERVAL 1
     df_result = results_model_2[1]
     df_result['Mean Return'] = df_result['Stock'].map(index_mean_returns_1)
     results_model_2[1] = df_result

     try:
         df_result = results_model_2[1]
         if 'Weight' in df_result.columns and 'Mean Return' in df_result.columns:
             sum_product = (df_result['Weight'] * df_result['Mean Return']).sum()
             results_model_2.append(sum_product * 100)
         else:
             print("The DataFrame does not contain both 'Peso' and 'Mean Return' columns")
     except Exception as e:
         print(f"Error: {e}")


     
     # Calculate the 2020 annual return for the S&P 500 index - Interval 1  (it's in percentage)
     index_return_1 = sum(index_mean_returns_1[title] * w0_1[title] for title in index_mean_returns_1 if title in w0_1) * 100

     return results_model_1, results_model_2, index_mean_returns_1, index_return_1
 
    
def calculate_portfolio_variance_mixture(covariance_matrix_1, w0_1, results_model_1, results_model_2):
     """
     Function to calculate the portfolio variances.

     Input:
     - covariance_matrix_1 (pd.DataFrame): Full covariance matrix.
     - w0_1 (dict): Dictionary with stock symbols and index weights.
     - results_model_1, results_model_2: model results.

     Output:
         - results_model_1, results_model_2: model results.
         - index_variance: variances of the S&P 500 index.
     """
     
     results_models = [results_model_1, results_model_2]
     
     # Calculate variance for the index
     tickers_index = list(w0_1.keys())
     weights_index = np.array(list(w0_1.values()))  # we only take the values of the weights
     cov_submatrix_index = covariance_matrix_1.loc[tickers_index, tickers_index]   # pull out the submatrix
     index_variance = np.dot(weights_index, np.dot(cov_submatrix_index, weights_index))

     # Update models 1 and 2 with portfolio variance
     for model in results_models:  # Models 1, 2
         df_results = model[1]
         tickers_ptf = df_results['Stock'].tolist()   # list of tickers
         weights_ptf = df_results['Weight'].values
         cov_submatrix_ptf = covariance_matrix_1.loc[tickers_ptf, tickers_ptf]
         ptf_var = np.dot(weights_ptf, np.dot(cov_submatrix_ptf, weights_ptf))
         model.append(ptf_var)

     return results_model_1, results_model_2, index_variance

def calculate_sharpe_ratios_mixture(results_model_1, results_model_2, index_return_1, index_variance_1):
     """
     Function to calculate the Sharpe Ratios of the portfolios.

     Input:
     - results_model_1, results_model_2: model results.
     - index_return_1: return of the S&P 500 index.
     - index_variance_1: variance of the S&P 500 index.

     Output:
         - results_model_1, results_model_2: model results.
         - SR_index: Sharpe ratio of the S&P 500 index.
     """
     
     results_models = [results_model_1, results_model_2]

     # Average closing rate for the 3-month Treasury throughout 2020(it's considered a good approximation of the annual risk-free rate)
     average_treasury_rate_2020 = 0.0033613095265288377

     # Calculate the daily risk-free rate for 2020(effective risk-free rate, 252 are the typical number of trading days in a year in the financial markets)
     risk_free_daily_2020 = (1 + average_treasury_rate_2020) ** (1 / 252) - 1

     # Calculate the annualized Sharpe ratio for 2020 (remember that the returns are dailys)
     SR_index = (index_return_1 / 100 - risk_free_daily_2020) / np.sqrt(index_variance_1) * np.sqrt(252)  

     # Calculate and store the Sharpe ratio for each model
     for model in results_models:
          SR_ptf = (model[4] / 100 - risk_free_daily_2020) / np.sqrt(model[5]) * np.sqrt(252)
          model.append(SR_ptf)

     return results_model_1, results_model_2, SR_index 
    
    
def calculate_tracking_error_out_mixture(interval_out, covariance_matrix_roll_out, w0_roll_out, results_model_1_roll_out, results_model_2_roll_out):
    """
    Calculate tracking error for out-of-samples test
    
    Input:
        -interval_out: Interval for out-of-sample test
        -covariance_matrix_roll_out: Covariance matrix of specified interval
        -w0_roll_out: Index weight for specified interval
        -results_model_1_roll_out, results_model_2_roll_out: Results of models out-of-sample
    
    Output:
        -track_error_1, track_error_2: Tracking error of each models for a specified time interval 
    """
    
    track_error_1=[]
    track_error_2=[]
        
    df_index_1_2 = pd.DataFrame(list(w0_roll_out.items()), columns=["Stock", "index_weight"])
    
    # MODEL 1
    df_merged_1 = results_model_1_roll_out[interval_out][1].merge(df_index_1_2, on='Stock', how='left')
    df_merged_1['diff']=df_merged_1['Weight']-df_merged_1['index_weight']

    tickers_ptf = df_merged_1['Stock'].tolist()
    diff_ptf = df_merged_1['diff'].values
    cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
    tracking_error_1 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
    
    track_error_1=tracking_error_1
    
    # MODEL 2
    df_merged_2 = results_model_2_roll_out[interval_out][1].merge(df_index_1_2, on='Stock', how='left')
    df_merged_2['diff']=df_merged_2['Weight']-df_merged_2['index_weight']

    tickers_ptf = df_merged_2['Stock'].tolist()
    diff_ptf = df_merged_2['diff'].values
    cov_submatrix_ptf = covariance_matrix_roll_out.loc[tickers_ptf, tickers_ptf]
    tracking_error_2 = np.dot(diff_ptf, np.dot(cov_submatrix_ptf, diff_ptf))
    
    track_error_2=tracking_error_2  

    return track_error_1, track_error_2


def calculate_tracking_ratio_out_mixture(intervals, market_caps_dict, market_caps_df_2020, results_model_1_roll, results_model_1_roll_out, results_model_2_roll, results_model_2_roll_out):
    """
    Calculate tracking ratios for out-of-samples test
    
    Input:
        - intervals: List of intervals for out-of-sample test
        - market_caps_dict: Dictionary opf market caps values for each interval
        - results_model_1_roll, results_model_2_roll: Results of the models for each intervals in-samples data 
        - results_model_1_roll_out, results_model_2_roll_out: Results of the models for each intervals out-of-samples data
    
    Output:
        -tracking_ratio_dict_1, tracking_ratio_dict_2: Tracking ratios for each models in each intervals
    """
    tracking_ratio_list_1={}
    tracking_ratio_list_2={}

    i=0

    for interval_out, market_caps_out in market_caps_dict.items():
        interval_out = intervals[i]
    
        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll = market_caps_df_2020.sum()
    
        # Compute initial weights (w0) based on stock market capitalization
        total_market_caps_roll_out = market_caps_out['Market Cap'].sum()
    
    
        # Calculate tracking ratios in-sample and out-of-sample
        tracking_ratio_model_1_roll_out, tracking_ratio_model_2_roll_out = calculate_tracking_ratio_mixture(results_model_1_roll, results_model_1_roll_out[interval_out], results_model_2_roll, results_model_2_roll_out[interval_out], market_caps_df_2020, market_caps_out, total_market_caps_roll, total_market_caps_roll_out)
        
        tracking_ratio_list_1[interval_out]=tracking_ratio_model_1_roll_out
        tracking_ratio_list_2[interval_out]=tracking_ratio_model_2_roll_out
        
        i+=1

    return tracking_ratio_list_1, tracking_ratio_list_2
    

def calculate_tracking_ratio_mixture(results_model_1, results_model_1_out, results_model_2, results_model_2_out, market_caps_df_1, market_caps_df_2, total_market_caps_1, total_market_caps_2):
    """
    Calculate tracking ratio in-sample and out-of-sample for different models.
    The tracking ratio (R0t) is calculated as the ratio between the performance of the reference index (S&P 500)
    and the performance of the tracking portfolio over a given period.
    Formula: R0t = ( ΣVit / ΣVi0 ) / ( ΣwjVjt / ΣwjVj0 )
    where:
    - ΣVit: sum of market values of all assets in the reference index (S&P 500) at time t.
    - ΣVi0: sum of market values of all assets in the reference index at the initial time (time 0).
    - ΣwjVjt: sum of market values of assets in the tracking portfolio at time t, weighted by their proportion.
    - ΣwjVj0: sum of market values of assets in the tracking portfolio at time 0, weighted by their initial proportion.
    
    Input:
        - results_model_1, results_model_2: Dictionaries with the results of the models for in-sample data.
        - results_model_1_out, results_model_2_out: Dictionaries with the results of the models for out-of-sample data.
        - market_caps_df_1: DataFrame with market capitalizations of assets of interval 1 (in samples data)
        - market_caps_df_2: DataFrame with market capitalizations of assets of interval 2 (out of samples data)
        - total_market_caps_1: Total market capitalizations of assets of interval 1 (in samples data)
        - total_market_caps_2: Total market capitalizations of assets of interval 2 (out of samples data)
    
    Output:
        - tracking_ratio_model_1, tracking_ratio_model_2: Tracking ratios values for models 
    """    

    df_merged_1 = results_model_1[1].merge(market_caps_df_1, left_on='Stock', right_index=True)
    df_merged_out = results_model_1_out[1].merge(market_caps_df_2, left_on='Stock', right_index=True)
    
    # Calculate the sum of the product Weight * Market Cap
    somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
    somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
    
    tracking_ratio_1=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
       
    df_merged_1 = results_model_2[1].merge(market_caps_df_1, left_on='Stock', right_index=True)
    df_merged_out = results_model_2_out[1].merge(market_caps_df_2, left_on='Stock', right_index=True)
    
    # Calculate the sum of the product Weight * Market Cap
    somma_peso_capitalizzazione_1 = (df_merged_1['Weight'] * df_merged_1['Market Cap']).sum()
    somma_peso_capitalizzazione_out = (df_merged_out['Weight'] * df_merged_out['Market Cap']).sum()
    
    tracking_ratio_2=(total_market_caps_2/total_market_caps_1)/(somma_peso_capitalizzazione_out/somma_peso_capitalizzazione_1)
        
    return tracking_ratio_1, tracking_ratio_2
