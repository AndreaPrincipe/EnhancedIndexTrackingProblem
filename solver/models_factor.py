import mosek.fusion as mf
import numpy as np
import pandas as pd
import time

def sqrt_matrix(covariance_matrix):
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    
    #the only case where we have negative eigen value is when we work with covariance matrix, but we have found that using the clipped matrix we can overcome this issue
    eigvals_clipped = np.clip(eigvals, 0, None)
    sqrt_eigvals = np.diag(np.sqrt(eigvals_clipped))
    covariance_sqrt = eigvecs @ sqrt_eigvals @ eigvecs.T

    if np.allclose(covariance_sqrt @ covariance_sqrt, covariance_matrix):
        return covariance_sqrt
    else:
        raise ValueError("Error: the matrix is not correct.")
        
def basic_tracking_mosek(mu, Cov_sqrt, market_caps_df, w0, sigma, TE, q, l=0.001, u=0.7):
    """
    Solves a mixed-integer second-order cone programming(MISOCP): 
    Selects a tracking portfolio with a constraint on the portfolio risk (sigma) 
    and on the tracking error (TE).
    
    Input:
        - mu: vector of the expected returns of assets.
        - Cov_sqrt: Square of covariance matrix stock returns.
        - market_caps_df (pd.DataFrame): DataFrame with market capitalizations of stocks.
        - w0 (dict): Dictionary of initial weights in the S&P 500 index.
        - sigma: porfolio risk limit.
        - TE: tracking error limit.
        - q: size tracking portfolio.
        - l: Lower bound for a stock weight (default: 0.001).
        - u: Upper bound for a stock weight (default: 1).

    Output:
        - objective_value (float): Optimal value of the objective function.
        - weights_df (pd.DataFrame): DataFrame with the weights of the selected stocks.
        - selected_tickers (list): Names of the selected stocks.
        - norm_diff (float): Norm of the difference between the selected portfolio weights and the index weights.
    """
    
    start_time = time.time()
    # Parametri di input
    n = Cov_sqrt.shape[0]
    tickers = market_caps_df.index.tolist()
    
    # Verifiche preliminari sulla matrice di covarianza
    if not np.allclose(Cov_sqrt, Cov_sqrt.T):
        raise ValueError("La matrice di covarianza deve essere simmetrica.")
    if q > n:
        raise ValueError("q deve essere minore o uguale al numero totale di azioni n.")
    
    # Creazione del modello
    model = mf.Model("index_tracking_model_1")
    
    # Variabili decisionali: pesi degli asset (x) e variabili binarie (y)
    x = model.variable("x", n, mf.Domain.inRange(0, 1))
    y = model.variable("y", n, mf.Domain.binary())
    
    # Funzione obiettivo: massimizzare il rendimento ponderato
    model.objective(mf.ObjectiveSense.Maximize, mf.Expr.dot(mu, x))
    
    w0_values = np.array([w0[market_caps_df.index[j]] for j in range(n)])
    # Calcola il prodotto tra Cov_sqrt e w0 usando NumPy
    constant_TE_np = Cov_sqrt @ w0_values  # Matrice n×n per vettore n×1

    # Converti in un'espressione Mosek
    constant_TE = mf.Expr.constTerm(constant_TE_np)
    
    model.constraint(mf.Expr.vstack(sigma, mf.Expr.mul(Cov_sqrt, x)), mf.Domain.inQCone())
    model.constraint(mf.Expr.vstack(TE, mf.Expr.sub(mf.Expr.mul(Cov_sqrt, x), constant_TE)), mf.Domain.inQCone())
    # Condizione sui pesi: somma dei pesi deve essere uguale a 1
    model.constraint(mf.Expr.sum(x), mf.Domain.equalsTo(1))
    
    # Cardinalità: selezionare esattamente `q` azioni
    model.constraint(mf.Expr.sum(y), mf.Domain.equalsTo(q))
    
    # Condizioni sui pesi: limiti superiori e inferiori legati alle variabili binarie
    model.constraint(mf.Expr.sub(x, mf.Expr.mul(l, y)), mf.Domain.greaterThan(0.0))
    model.constraint(mf.Expr.sub(x, mf.Expr.mul(u, y)), mf.Domain.lessThan(0.0))
    
    model.solve()
    
    # Check model status
    status = model.getProblemStatus(mf.SolutionType.Default)
    sol_status = model.getPrimalSolutionStatus(mf.SolutionType.Default)
    
    # Check if the solution status is optimal
    if sol_status == mf.SolutionStatus.Optimal:
        print("Optimal solution found.")
    
    # Check if only a feasible integer solution was found (not necessarily optimal)
    elif status == mf.ProblemStatus.PrimalFeasible:
        print("Feasible integer solution found (not necessarily optimal).")
    
    # Check if the problem is primal infeasible
    elif status == mf.ProblemStatus.PrimalInfeasible:
        print("The problem is primal infeasible.")
    
    # Check if the problem is dual infeasible
    elif status == mf.ProblemStatus.DualInfeasible:
        print("The problem is dual infeasible.")
    
    # Check if the problem is primal infeasible or unbounded
    elif status == mf.ProblemStatus.PrimalInfeasibleOrUnbounded:
        print("The problem is primal infeasible or unbounded.")
    
    # Handle any unknown status
    else:
        print(f"Unknown status: {status}")



    # Extract selected stocks
    y_values = y.level()  # Get the values of the binary variables
    y_selected = [j for j in range(n) if y_values[j] > 0.5]  # Select stocks where y_j = 1
    selected_tickers = [tickers[j] for j in y_selected]
    
    # Extract weights
    x_values = x.level()
    weights = {tickers[j]: x_values[j] for j in y_selected}
    
    # Create DataFrame of weights
    weights_df = pd.DataFrame(weights.items(), columns=["Stock", "Weight"])
    
    # Optional sorting by descending weight
    weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)
    
    # Compute the norm of the difference
    differences = [weights[stock] - w0[stock] for stock in weights]  # Assume w0 contains all keys, otherwise use .get(stock, 0) with error handling
    norm_diff = np.linalg.norm(differences)
    
    # Optimal objective function value
    objective_value = model.primalObjValue()
    
    end_time = time.time()
    total_time = end_time - start_time

    return objective_value, weights_df, selected_tickers, norm_diff, total_time




def robust_tracking_mosek(mu_0, gamma_down, F, V0, rho, G, d_up, market_caps_df, w0, sigma, TE, q, prev_solution,l_bound=0.001, u_bound=0.7):
    """
    Solves a mixed-integer second-order cone programming(MISOCP): 
    Selects a tracking portfolio with a robust constraint on the portfolio risk (sigma) 
    and on the tracking error (TE).
    
    Input:
        - mu_0: nominal vector of the expected returns of assets.
        - gamma_down: lower bound of interval uncertainity set for the expected return.
        - F: covariance matrix of returns of the factors that drive the market (mxm).
        - V0: nominal matrix of factor loadings of the n assets (mxn).
        - rho: radius of ellipsoid uncertainity set for V.
        - G: denotes the coordinate system(for V) that may not be perpendicular (is positive definite).
        - d_up: upper bound of interval uncertainity set for d_i (D=diag(d) is the covariance matrix of residual returns).
        - market_caps_df (pd.DataFrame): DataFrame with market capitalizations of stocks.
        - w0 (dict): Dictionary of initial weights in the S&P 500 index.
        - sigma: porfolio risk limit.
        - TE: tracking error limit.
        - q: size tracking portfolio.
        - l: Lower bound for a stock weight (default: 0.001).
        - up: Upper bound for a stock weight (default: 1).

    Output:
        - objective_value (float): Optimal value of the objective function.
        - weights_df (pd.DataFrame): DataFrame with the weights of the selected stocks.
        - selected_tickers (list): Names of the selected stocks.
        - norm_diff (float): Norm of the difference between the selected portfolio weights and the index weights.
    """
    
    start_time = time.time()
    n = mu_0.shape[0]
    m = V0.shape[0]
    tickers = market_caps_df.index.tolist()

    # Create the optimization model
    model = mf.Model("index_tracking_model_robust_1")

    # Decision variables
    x = model.variable("weights", n, mf.Domain.inRange(0, 1))
    delta = model.variable("delta", mf.Domain.greaterThan(0))
    nu = model.variable("nu", mf.Domain.greaterThan(0))
    tao = model.variable("tao", mf.Domain.greaterThan(0))
    s = model.variable("s", m, mf.Domain.greaterThan(0))
    y = model.variable("y", n, mf.Domain.binary())
    z_plus = model.variable("z_plus", n, mf.Domain.greaterThan(0))
    z_minus = model.variable("z_minus", n, mf.Domain.greaterThan(0))
    u = model.variable("u", m)
    w = model.variable("w", m)
    l = model.variable("l", mf.Domain.greaterThan(0))
    z = model.variable("z", mf.Domain.greaterThan(0))
    
    # Robust objective function
    coeff = [mu_0[i][0] + gamma_down[i][0] for i in range(n)]
    model.objective("obj", mf.ObjectiveSense.Maximize, mf.Expr.dot(coeff, x))

    # Robust risk constraint
    sqrt_F = sqrt_matrix(F)
    K = sqrt_F @ np.linalg.inv(G) @ sqrt_F
    eigenvalue, P = np.linalg.eigh(K)  #K is symmetric(since F and G are) so we can use the command 'eigh'
    #teta_matrix= np.diag(eigenvalue)
    teta = eigenvalue
    
    #constraint 21
    coeff_u = P.T @ sqrt_F @ V0  
    model.constraint(mf.Expr.sub(u, mf.Expr.mul(coeff_u,x)), mf.Domain.equalsTo(0.0))
        
    #constraint 22
    model.constraint(mf.Expr.sub(w, mf.Expr.mul(coeff_u, mf.Expr.sub(z_plus, z_minus))), mf.Domain.equalsTo(0.0))

    #constraint 23
    w0_values = np.array([w0[market_caps_df.index[j]] for j in range(n)])  #vector with the banchmark weights in the same order of the dictionary
    model.constraint(mf.Expr.sub(mf.Expr.sub(z_plus, z_minus), mf.Expr.sub(x, w0_values)), mf.Domain.equalsTo(0.0))

    #constraint 24
    fusion_1 = mf.Expr.vstack(
        mf.Expr.mul(2.0, mf.Expr.dot(rho.ravel(), x)), 
        mf.Expr.add(mf.Expr.sub(tao, nu), mf.Expr.sum(s))
    )
    model.constraint(mf.Expr.vstack(mf.Expr.sub(mf.Expr.add(tao, nu), mf.Expr.sum(s)), fusion_1), mf.Domain.inQCone())

    #constraint 25
    for j in range(m):
        fusion_2 = mf.Expr.vstack(
            mf.Expr.mul(2.0, u.index(j)), 
            mf.Expr.sub(mf.Expr.sub(nu, mf.Expr.mul(tao,teta[j])), s.index(j))
        )
        model.constraint(mf.Expr.vstack(mf.Expr.add(mf.Expr.sub(nu, mf.Expr.mul(tao,teta[j])), s.index(j)), fusion_2), mf.Domain.inQCone())
    
    #constraint 26
    fusion_3 = mf.Expr.vstack(
        mf.Expr.mul(2.0, mf.Expr.dot(rho.ravel(), mf.Expr.add(z_plus,z_minus))), 
        mf.Expr.add(mf.Expr.sub(tao, l), mf.Expr.sum(s))
    )
    model.constraint(mf.Expr.vstack(mf.Expr.sub(mf.Expr.add(tao, l), mf.Expr.sum(s)), fusion_3), mf.Domain.inQCone())
    
    #constraint 27
    for j in range(m):
        fusion_4 = mf.Expr.vstack(
            mf.Expr.mul(2.0, w.index(j)), 
            mf.Expr.sub(mf.Expr.sub(l, mf.Expr.mul(tao,teta[j])), s.index(j))
        )
        model.constraint(mf.Expr.vstack(mf.Expr.add(mf.Expr.sub(l, mf.Expr.mul(tao,teta[j])), s.index(j)), fusion_4), mf.Domain.inQCone())
   
    #constraint 28
    #d_up_vector = np.full(n, d_up)
    #D_up = np.diag(d_up_vector)
    #D_up_sqrt = sqrt_matrix(D_up) #it's still diagonal
    D_up_sqrt = np.diag(np.sqrt(d_up.flatten()))
    fusion_5 = mf.Expr.vstack(mf.Expr.mul(2.0, mf.Expr.mul(D_up_sqrt, x)), mf.Expr.sub(1.0, delta))
    model.constraint(mf.Expr.vstack(mf.Expr.add(1.0, delta), fusion_5), mf.Domain.inQCone())
    
    #constraint 29
    model.constraint(mf.Expr.sub(mf.Expr.add(nu, delta), sigma**2), mf.Domain.lessThan(0.0))
    
    #constraint 30
    fusion_6 = mf.Expr.vstack(mf.Expr.mul(2.0, mf.Expr.mul(D_up_sqrt, mf.Expr.sub(z_plus, z_minus))), mf.Expr.sub(1.0, z))
    model.constraint(mf.Expr.vstack(mf.Expr.add(1.0, z), fusion_6), mf.Domain.inQCone())
    
    #constraint 31
    model.constraint(mf.Expr.sub(mf.Expr.add(l, z), TE**2), mf.Domain.lessThan(0.0))
    
    #constraint 32
    max_eigen = np.max(eigenvalue)
    model.constraint(mf.Expr.sub(nu, mf.Expr.mul(tao, max_eigen)), mf.Domain.greaterThan(0.0))
    
    #constraint 33
    model.constraint(mf.Expr.sub(l, mf.Expr.mul(tao, max_eigen)), mf.Domain.greaterThan(0.0))

    # weight condition-34
    model.constraint(mf.Expr.sum(x), mf.Domain.equalsTo(1))

    # Cardinality: select exactly `q` stocks-35
    model.constraint(mf.Expr.sum(y), mf.Domain.equalsTo(q))

    # Weight bounds-36
    model.constraint(mf.Expr.sub(x, mf.Expr.mul(l_bound, y)), mf.Domain.greaterThan(0.0))
    model.constraint(mf.Expr.sub(x, mf.Expr.mul(u_bound, y)), mf.Domain.lessThan(0.0))
    
    if prev_solution is not None:
        prev_x = prev_solution.get("x", None)
        prev_y = prev_solution.get("y", None)
    
        if prev_x is not None:
            x.setLevel(prev_x)
        if prev_y is not None:
            y.setLevel(prev_y)
            
            
    # Imposta parametri iniziali
    model.setSolverParam("mioTolRelGap", 1e-3)   # gap stretto iniziale
    model.setSolverParam("mioMaxTime", 120.0)     # pre-solve rapido
    
    # Primo tentativo veloce
    model.solve()
    
    # Check model status
    status = model.getProblemStatus(mf.SolutionType.Default)
    sol_status = model.getPrimalSolutionStatus(mf.SolutionType.Default)
    
    sol_status = model.getPrimalSolutionStatus(mf.SolutionType.Default)
    if sol_status not in [mf.SolutionStatus.Optimal]:
        model.setSolverParam("mioTolRelGap", 0.03)
        model.setSolverParam("mioHeuristicLevel", 3)
        #model.setSolverParam("mioNodeSelection", "best")
        model.setSolverParam("mioCutSelectionLevel", 1)
        model.setSolverParam("mioMaxTime", 600.0) 
        model.acceptedSolutionStatus(mf.AccSolutionStatus.Feasible)
        model.solve()
    
    # Check if the solution status is optimal
    if sol_status == mf.SolutionStatus.Optimal:
        print("Optimal solution found.")
    
    # Check if only a feasible integer solution was found (not necessarily optimal)
    elif status == mf.ProblemStatus.PrimalFeasible:
        print("Feasible integer solution found (not necessarily optimal).")
    
    # Check if the problem is primal infeasible
    elif status == mf.ProblemStatus.PrimalInfeasible:
        print("The problem is primal infeasible.")
    
    # Check if the problem is dual infeasible
    elif status == mf.ProblemStatus.DualInfeasible:
        print("The problem is dual infeasible.")
    
    # Check if the problem is primal infeasible or unbounded
    elif status == mf.ProblemStatus.PrimalInfeasibleOrUnbounded:
        print("The problem is primal infeasible or unbounded.")
    
    # Handle any unknown status
    else:
        print(f"Unknown status: {status}")
        

    # Extract selected stocks
    solution_dict = {
        "x": x.level().tolist(),
        "y": y.level().tolist()
    }
    y_values = y.level()  # Get the values of the binary variables
    y_selected = [j for j in range(n) if y_values[j] > 0.5]  # Select stocks where y_j = 1
    selected_tickers = [tickers[j] for j in y_selected]
    
    # Extract weights
    x_values = x.level()
    weights = {tickers[j]: x_values[j] for j in y_selected}
    
    # Create DataFrame of weights
    weights_df = pd.DataFrame(weights.items(), columns=["Stock", "Weight"])
    
    # Optional sorting by descending weight
    weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)
    
    # Compute the norm of the difference
    differences = [weights[stock] - w0[stock] for stock in weights]  # Assume w0 contains all keys, otherwise use .get(stock, 0) with error handling
    norm_diff = np.linalg.norm(differences)
    
    # Optimal objective function value
    objective_value = model.primalObjValue()
    
    end_time = time.time()
    total_time = end_time - start_time

    return objective_value, weights_df, selected_tickers, norm_diff, total_time, solution_dict
