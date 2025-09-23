# Enhanced Index Tracking Problem



## Overview of the models
This repository implements two formulations of the Enhanced Index Tracking Problem (EITP):

### 1. Factor-based Robust Model
- Built on **Fama–French factor models** (both 3-factor and 5-factor variants).  
- Robustness introduced through **uncertainty sets** on:  
  - Expected returns  
  - Factor loadings matrix  
- Formulated as a **Mixed-Integer Second-Order Cone Program (MISOCP)**, solved with **Mosek Fusion**.  
- Includes **cardinality constraints** to enforce portfolio sparsity.  
- Evaluation:  
  - **Static out-of-sample tests**  
  - **Rolling-window analysis** for dynamic performance assessment  

### 2. Mixture-based Robust Model
- Models asset returns using a **Gaussian Mixture distribution**.  
- Incorporates uncertainty in mixture weights via **ϕ-divergences**.  
- Reformulated using **Lagrangian duality**, leading to a tractable optimization problem solved with **Gurobi**.  
- Evaluation:  
  - **Static out-of-sample tests**  
  - (Rolling-window analysis was computationally infeasible at scale)

For a detailed description of the models, see the following PDFs:

- [Factor-based Robust Model](Factor-based_robust_index_tracking.pdf)
- [Mixture-based Robust Model](Robust_enhanced_index_tracking_problem_with_mixture_of_distributions.pdf)


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
│   ├── models_factor.py         
│   └── plot_functions.py                
├── results/                              
│   ├── results_factor
│   ├── results_mixture  
│
```
## Requirements
To correctly run the project, ensure that the following dependencies are installed:
```plaintext
-**gurobipy==11.0.3**
-**mosek==11.0.11**
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
The Python version used is 3.12.4. Furthermore, an active license is required for Gurobi and Mosek extensions.

## Execution - Usage Instructions

The project can be executed through two main scripts, one for each formulation of the Enhanced Index Tracking Problem. Running `python main_factor.py` starts the factor-based robust model, which uses the Mosek Fusion API, while `python main_mixture.py` runs the mixture-based robust model implemented with Gurobi.  

All the required input data are already provided in the `data/` directory, including market capitalization files, stock time series, and Fama–French factors, so no additional preprocessing is needed before execution. When running the scripts, any plots and visualizations of portfolio performance are saved inside the `results/` folder, in the corresponding subfolder (`results_factor` or `results_mixture`), **but only if `save_figures=True` is set in the main scripts**.

The implementation has been tested on a machine with 16 GB of RAM.
