# Enhanced Index Tracking Problem

## Overview of the models

## Project Structure
```plaintext
EnhancedIndexTrackingProblem/
├── README.md                          
├── requirements.txt                    
├── main_mixture.py
├── main_factor.py                            
├── Factor-based_robust_index_tracking.pdf
├── Robust_enhanced_index_tracking_problem_with_mixture_of_distributions.pdf
├── data/                                
│   ├── market_caps_df_2020.csv           
│   ├── market_caps_df_2021.csv          
│   ├── data_stocks_filtered.pkl         
│   ├── sp500_companies.csv              
│   ├── market_caps_dict.pkl
│   ├── F-F_Research_Data_5_Factors_2x3_daily.csv        
├── solver/                              
│   ├── __init__.py                      
│   ├── functions.py   
│   ├── models_mixture.py
│   ├── mosek_fusion_models.py         
│   └── plot_functions_robust.py                
├── results/                              
│   ├── results_factor
│   ├── results_mixture  
│
```
## Requirements
To correctly run the project, ensure that the following dependencies are installed:
```plaintext
-**gurobipy==11.0.3**
-**matplotlib==3.5.1**
-**numpy==1.20.3**
-**pandas==1.4.1**
-**yfinance==0.2.51**
-**pickle**
-**io**
-**PIL**
-**copy**
-**itertools**
-**pathlib**
```
The Python version used is 3.12.4. Furthermore, an active license is required for the Gurobi extension.

## Execution - Usage Instructions
