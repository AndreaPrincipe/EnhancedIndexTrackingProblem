import os
os.environ["OMP_NUM_THREADS"] = "1"  # To avoid problem with KMeans on Windows
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import norm
import pandas as pd
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def adaptive_nodes(f, a, b, tol=1e-3, max_pts=1000):
    """
    f       : univariate function (e.g., norm.pdf or norm.cdf)
    a, b    : interval boundaries
    tol     : interpolation error tolerance
    max_pts : maximum number of nodes to prevent explosion
    """
    
    nodes = [a, b]
    
    def refine(x0, x1):
        xm = 0.5*(x0 + x1)
        y0, y1, ym = f(x0), f(x1), f(xm)
        y_lin = 0.5*(y0 + y1)
        if abs(ym - y_lin) > tol and len(nodes) < max_pts:
            refine(x0, xm)
            nodes.append(xm)
            refine(xm, x1)
            
    refine(a, b)
    return sorted(nodes)

def kmeans_initialization(X,d):
    """
    K-means based initialization for Gaussian Mixture Models.

    X : data matrix with shape (n_day, n_stocks+1)
    d : number of clusters/components
    """

    n = X.shape[1] # number of features
    
    # 1. K-means for initial cluster centers (means)
    kmeans = KMeans(n_clusters=d, n_init=20, random_state=1)  #Among the tested values (1,10,42), 1 provides the lowest inertia, so we selected it for our final initialization.
    labels = kmeans.fit_predict(X)
    mu_init = kmeans.cluster_centers_
    
    # 2. Lambda Initialization as proportion of points inside each cluster
    lambda_init = np.array([(labels == k).sum() for k in range(d)]) / len(X)
    
    # 3. Covariance initialization for each cluster
    cov_init = np.zeros((d, n, n))
    
    for k in range(d):
        X_k = X[labels == k]
        cov = np.cov(X_k, rowvar=False) + 1e-5 * np.eye(n)  # add regularization
        cov_init[k]= cov
    
    # Outputs:
    # - mu_init: array of shape (d, n_features)
    # - lambda_init: array of shape (d,)
    # - cov_init: array of shape (d, n_features, n_features)
    
    return mu_init, lambda_init, cov_init


def sklearn_function(X, new_data_stocks_interval_1, mu_init, cov_init, lambda_init, d):
    """
    Fit Gaussian Mixture Model with EM algorithm using custom initialization.

    X : data matrix
    mu_init : initial means
    cov_init : initial covariance matrices
    lambda_init : initial mixture weights
    d : number of mixture components
    """
    
    gmm = GaussianMixture(
    n_components=d,
    covariance_type='full',  
    max_iter=100,
    init_params='random',  
    random_state=0
    )
    
    # Set initial parameters
    gmm.weights_init = lambda_init
    gmm.means_init = mu_init
    gmm.precisions_init = np.linalg.inv(cov_init)  # precision = inverse covariance
    
    # Fit model with EM
    gmm.fit(X)
    n_iterations = gmm.n_iter_
    print(f"Numero di iterazioni EM: {n_iterations}")
    
    # Final estimates
    lambda_final = gmm.weights_       # array of shape (n_components,)
    mu_final = gmm.means_             # array of shape (n_components, n_features)
    cov_final = gmm.covariances_      # array of shape (n_components, n_features, n_features)

    return mu_final, cov_final, lambda_final

def model_EIT1(lamb, mu, cov, market_caps_df, w0_1, new_data_stocks_interval_1):
    start_time = time.time()
    n = mu.shape[1] -1 #number of assets
    d = len(lamb) # components number
    c_i = 0.01 # It is the unit transaction cost for the stock, it's always the same
    k=1.9841e-4 # To achieve an annual return of 5%
    x_vals = adaptive_nodes(norm.pdf, -10, 10, 1e-4)
    y_vals_pdf = [norm.pdf(x) for x in x_vals]
    y_vals_cdf = [norm.cdf(x) for x in x_vals]
    pdf_ub = float(np.max(y_vals_pdf))
    
    # Bounds for volatility and variance
    bounds_sq = n + 1
    sigma2_ub = []
    for i in range(d):
        # calcola il massimo autovalore (essendo Cov PSD, usiamo eigvalsh)
        lambda_max = np.linalg.eigvalsh(cov[i])[-1]
        sigma2_ub.append(lambda_max * bounds_sq)
    volatility_ub = [ np.sqrt(u) for u in sigma2_ub ]
    vol_lb = [1e-5]*d
    
    # Bounds for auxiliary variable L; here we fix the values based on empirical findings
    Llb = [5e-05, 0.0001, 5e-05]
    Lub= [5e-05, 0.0001, 5e-05]
    
    # Bounds for x_tilde(weights)
    # x_tilde: weights + dummy = n+1 vars
    lbs = [0]*n + [-1]
    ubs = [1]*n + [-1]
    
    # Create the optimization model
    model = gp.Model("mixture_EITP_1")
    
    # Allow the model to handle non-convex quadratic constraints
    model.Params.NonConvex = 2
    
    # Decision variables
    x_tilde = [model.addVar(lb=lbs[j],ub=ubs[j], name=f"x_tilde[{j}]") for j in range(n+1)]
    volatility = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=vol_lb,ub=1, name="volatility") #volatilityub_dict
    L          = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=Llb,ub=Lub, name="L") #lb=Llb_dict,ub=Lub_dict
    cdf_approx = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=0,ub=1, name="cdf_approx")
    pdf_approx = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=0, ub=pdf_ub, name="pdf_approx")
    y = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="abs_value")
    #w = model.addVars(range(d),lb={i: Llb[i] * vol_lb[i] for i in range(d)},   # inizialmente 0 come vol LB
    #              ub={i: Lub[i] * volatility_ub[i] for i in range(d)},
    #              name="w")
    '''
    #McCormick for a bilinear product
    for i in range(d):
        xL, xU = Llb[i],Lub[i] #Llb_dict[i], Lub_dict[i]
        yL, yU = vol_lb[i], volatility_ub[i]   # volatility.lb = eps (non può essere proprio 0), volatilityub_dict[i]
        
        # McCormick envelope:
        model.addConstr(w[i] >= xL*volatility[i] + yL*L[i] - xL*yL,
                        name=f"Mc1_{i}")
        model.addConstr(w[i] >= xU*volatility[i] + yU*L[i] - xU*yU,
                        name=f"Mc2_{i}")
        model.addConstr(w[i] <= xL*volatility[i] + yU*L[i] - xL*yU,
                        name=f"Mc3_{i}") 
        model.addConstr(w[i] <= xU*volatility[i] + yL*L[i] - xU*yL,
                        name=f"Mc4_{i}")
    '''
    # precompute nonzero indices per cov
    nz_list = [list(zip(*np.nonzero(cov[i]))) for i in range(d)]
    
    # Empty list to store mean values for the different components
    nu = []
    
    # Empty list to store R1 values for the different components
    R1 = []
    
    # R1 definition
    for i in range(d):
        quad_expr = gp.QuadExpr()
        for j, s in nz_list[i]:
            quad_expr.add(x_tilde[j] * x_tilde[s] * cov[i][j, s])
        
        # Volatility
        model.addQConstr(quad_expr <= volatility[i]*volatility[i], name=f"volatility_qc_{i}")
        
        # Mean
        nu.append(gp.quicksum(x_tilde[l]*mu[i][l] for l in range(n+1)))
        
        #Relationship (k - nu) = L * volatility
        #model.addConstr((k - nu[i]) == w[i], name=f"lin_bilinear_{i}")
        model.addConstr((k - nu[i]) == L[i] * volatility[i], name=f"lin_bilinear_{i}")
        
        # Piecewise linear approximation (PWL) for the PDF and CDF of the normal distribution
        model.addGenConstrPWL(
        xvar=L[i],
        yvar=pdf_approx[i],
        xpts=x_vals,
        ypts=y_vals_pdf,
        name=f"pdf_pwl_{i}"
        )
        model.addGenConstrPWL(
        xvar=L[i],
        yvar=cdf_approx[i],
        xpts=x_vals,
        ypts=y_vals_cdf,
        name=f"cdf_pwl_{i}"
        )
        
        #R1 calculation
        R1.append((k-nu[i])*cdf_approx[i]+volatility[i]*pdf_approx[i])
        
    # Objective function
    obj = gp.quicksum(lamb[j]*R1[j] for j in range(d))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Absolute value for the transaction costs
    for i in range(n):
        model.addConstr(y[i] >= x_tilde[i] , name=f"abs_pos_{i}")
        model.addConstr(y[i] >= -x_tilde[i] , name=f"abs_neg_{i}")
    
    # Balancing constraint
    model.addConstr((gp.quicksum(x_tilde[i] for i in range(n))) + gp.quicksum(y[i]*c_i for i in range(n)) == 1.0)
    
    # Parameters
    model.setParam('ScaleFlag',     3)    # scaling aggressivo per QCP 3
    model.setParam('Presolve',      2)    # presolve più intenso 2
    model.setParam('Aggregate',     2)    # aggrega vincoli simili 2
    model.setParam('PreMIQCPForm', 1)     # per una riformulazione più efficiente dei coni quadrati prima di entrare in barrier

    
    # ===== Looser tolerances =====
    model.setParam('BarQCPConvTol', 1e-6)  # barrier conv tol
    model.setParam('FeasibilityTol',5e-6)  # toll. fattibilità
    model.setParam('OptimalityTol', 1e-6)  # toll. ottimalità
    model.setParam("BarConvTol", 1e-5) #new with L fixed
    
    # ===== Gap =====
    model.setParam('MIPGap',        1e-4)  # gap relativo
    
    # ===== MIP Strategies =====
    model.setParam('MIPFocus',      1)     # favorisce soluzioni iniziali  1
    model.setParam('Heuristics',    0.5)   # euristiche ridotte   0.6
    model.setParam('Cuts',          1)     # tagli disabilitati   2
    model.setParam('VarBranch',     2)     # branching rule leggera 1
    model.setParam('ConcurrentMIP', 1)     # prova 2 configurazioni in parallelo 1
    
    # ===== Increased numerical precision =====
    model.setParam('NumericFocus', 2)     # 0=default, 1=più veloce, 2=più preciso
    
    # Time limit
    model.setParam("TimeLimit", 3600)
    
    # ===== Parallelism =====
    model.setParam('Threads',       8) 
    
    # Optimize
    model.optimize()
    
    
    if model.Status == GRB.OPTIMAL:
        print("\nSolution quality:")
        model.printQuality()
    
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Model status {model.Status}: infeasible or unbounded")

    tickers = list(new_data_stocks_interval_1.keys())

    x_selected = [j for j in range(n) if x_tilde[j].X > 1e-4]  
    selected_tickers = [tickers[j] for j in x_selected]

    weights = {
        tickers[i]: x_tilde[i].X
        for i in x_selected
    }
    
    # Create DataFrame of weights
    weights_df = pd.DataFrame(weights.items(), columns=["Stock", "Weight"])

    # Optional sorting by descending weight
    weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)

    # Compute the norm of the difference
    differences = [weights[stock] - w0_1.get(stock, 0) for stock in weights]
    norm_diff = np.linalg.norm(differences)

    # Optimal objective function value
    objective_value = model.objVal
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Model status code: {model.Status}")


    return objective_value, weights_df, selected_tickers, norm_diff, total_time




def model_mixture_DEIT1(lamb, mu, cov, market_caps_df, w0_1, new_data_stocks_interval_1):
    
    start_time = time.time()
    n = mu.shape[1] -1 # Number of assets
    d = len(lamb) # Components number
    c_i = 0.01 # It is the unit transaction cost for the stock, it's always the same
    k=1.9841e-4 # To achieve an annual return of 5%
    rho = 0.05
    eps = 1e-6
    x_vals = adaptive_nodes(norm.pdf, -10, 10, 1e-4)
    y_vals_pdf = [norm.pdf(x) for x in x_vals]
    y_vals_cdf = [norm.cdf(x) for x in x_vals]
    pdf_ub = float(np.max(y_vals_pdf))
    
    # Bounds for volatility and variance
    bounds_sq = n + 1
    sigma2_ub = []
    for i in range(d):
        # Compute the largest eigenvalue (Cov is PSD, so we use eigvalsh)
        lambda_max = np.linalg.eigvalsh(cov[i])[-1]
        sigma2_ub.append(lambda_max * bounds_sq)
    volatility_ub = [ np.sqrt(u) for u in sigma2_ub ]
    vol_lb = [1e-5]*d
    
    # Bounds for mean
    nu_lb, nu_ub = [], []
    for i in range(d):
        pos = sum(max(0, mu[i][j]) * 1 for j in range(n)) + mu[i][-1]*(-1)
        neg = sum(min(0, mu[i][j]) * 1 for j in range(n)) + mu[i][-1]*(-1)
        nu_ub.append(pos)
        nu_lb.append(neg)
                
    # Bound for auxiliary variable L, but for the fix-and-relax strategy we don't use these bounds.
    L_lb = []
    L_ub = []
    for i in range(d):
        nu_l, nu_u   = nu_lb[i], nu_ub[i]
        vol_l, vol_u = vol_lb[i], volatility_ub[i]
        c1 = (k - nu_l)/vol_l
        c2 = (k - nu_l)/vol_u
        c3 = (k - nu_u)/vol_l
        c4 = (k - nu_u)/vol_u
        L_lb.append(min(c1, c2, c3, c4))
        L_ub.append(max(c1, c2, c3, c4))
    Lub_dict = dict(enumerate(L_ub))
    Llb_dict = dict(enumerate(L_lb))
    
    # Bounds for x_tilde
    # x_tilde: weights + dummy = n+1 vars
    lbs = [0]*n + [-1]
    ubs = [1]*n + [-1]
    
    # Bounds for auxiliary variable L; here we fix the values based on empirical findings
    Llb = [1.5,2.5,1.64]
    Lub= [1.5,2.5,1.64]
    
    # Create the optimization model
    model = gp.Model("mixture_DEITP")
    
    # Allow the model to handle non-convex quadratic constraints
    model.Params.NonConvex = 2
    
    # Decision variables
    x_tilde = [model.addVar(lb=lbs[j],ub=ubs[j], name=f"x_tilde[{j}]") for j in range(n+1)]
    volatility = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=vol_lb,ub=1, name="volatility") #volatilityub_dict
    exp1       = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=-5, ub=5, name="exp1")  #-10,10
    L          = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=Llb,ub=Lub, name="L") #lb=Llb_dict,ub=Lub_dict
    cdf_approx = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=0,ub=1, name="cdf_approx")
    pdf_approx = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=0, ub=pdf_ub, name="pdf_approx")
    exp_aux    = model.addVars(range(d), vtype=GRB.CONTINUOUS, lb=-3, ub=3, name="exp_aux")
    
    teta = model.addVar(vtype=GRB.CONTINUOUS, lb=-3, ub=3, name="dual_eq") #-5,5
    zeta = model.addVar(vtype=GRB.CONTINUOUS, lb=eps, ub=3, name="dual_dis") #ub:5
    y = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0, name="abs_value")
    #w = model.addVars(range(d),lb={i: Llb[i] * vol_lb[i] for i in range(d)},
    #              ub={i: Lub[i] * volatility_ub[i] for i in range(d)},
    #              name="w")

    # Precompute the nonzero indices for each covariance matrix
    nz_list = [list(zip(*np.nonzero(cov[i]))) for i in range(d)]
    # Empty list to store mean values for the different components
    nu = []
    # Empty list to store R1 values for the different components
    R1 = []
    
    '''
    for i in range(d):
        xL, xU = Llb[i],Lub[i] #Llb_dict[i], Lub_dict[i]
        yL, yU = vol_lb[i], volatility_ub[i]   # volatility.lb = eps (non può essere proprio 0), volatilityub_dict[i]
        
        # McCormick envelope:
        model.addConstr(w[i] >= xL*volatility[i] + yL*L[i] - xL*yL,
                        name=f"Mc1_{i}")
        model.addConstr(w[i] >= xU*volatility[i] + yU*L[i] - xU*yU,
                        name=f"Mc2_{i}")
        model.addConstr(w[i] <= xL*volatility[i] + yU*L[i] - xL*yU,
                        name=f"Mc3_{i}") 
        model.addConstr(w[i] <= xU*volatility[i] + yL*L[i] - xU*yL,
                        name=f"Mc4_{i}")
    '''
    
    # R1 definition
    for i in range(d):
        quad_expr = gp.QuadExpr()
        for j, s in nz_list[i]:
            quad_expr.add(x_tilde[j] * x_tilde[s] * cov[i][j, s])
        
        #Volatility
        model.addQConstr(quad_expr <= volatility[i]*volatility[i], name=f"volatility_qc_{i}")
        
        # Mean
        nu.append(gp.quicksum(x_tilde[l]*mu[i][l] for l in range(n+1)))
        
        #model.addQConstr((k - nu[i]) == w[i])
        model.addConstr((k - nu[i]) == L[i] * volatility[i], name=f"lin_bilinear_{i}")
        
        # Piecewise linear approximation (PWL) for the PDF and CDF of the normal distribution
        model.addGenConstrPWL(
        xvar=L[i],
        yvar=pdf_approx[i],
        xpts=x_vals,
        ypts=y_vals_pdf,
        name=f"pdf_pwl_{i}"
        )
        model.addGenConstrPWL(
        xvar=L[i],
        yvar=cdf_approx[i],
        xpts=x_vals,
        ypts=y_vals_cdf,
        name=f"cdf_pwl_{i}"
        )
        
        #R1 calculation
        R1.append((k-nu[i])*cdf_approx[i]+volatility[i]*pdf_approx[i])
    
    
    # Exponential constraint
    n_pieces = 40  # Number of points for the piecewise linear approximation
    x_min, x_max = -3, 3  # Realistic bounds for exp_aux[i]
    
    # Generate points over the interval for PWL
    x_vals_exp = np.linspace(x_min, x_max, n_pieces)
    y_vals_exp = np.exp(x_vals_exp)  # Corresponding exponential values
    
    for i in range(d):
        # Define the bilinear constraint q = exp_aux[i] * zeta - R1[i] + teta
        q = exp_aux[i] * zeta - R1[i] + teta
        model.addQConstr(q == 0, name=f"bilinear_{i}")
        
        # Add piecewise linear approximation of the exponential function
        model.addGenConstrPWL(
            exp_aux[i], exp1[i],
            x_vals_exp.tolist(), y_vals_exp.tolist(),
            name=f"pwl_exp_{i}"
        )
    
    # Objective function
    obj = teta + zeta * rho + zeta*gp.quicksum(lamb[j]*(exp1[j]-1) for j in range(d))
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Absolute value for the transaction costs
    for i in range(n):
        model.addConstr(y[i] >= x_tilde[i] , name=f"abs_pos_{i}")
        model.addConstr(y[i] >= -x_tilde[i] , name=f"abs_neg_{i}")
    
    # Balancing constraint
    model.addConstr((gp.quicksum(x_tilde[i] for i in range(n))) + gp.quicksum(y[i]*c_i for i in range(n)) == 1.0)
  
    # ===== Scale and presolve =====
    model.setParam('ScaleFlag',     3)    # scaling aggressivo per QCP 3
    model.setParam('Presolve',      2)    # presolve più intenso 2
    model.setParam('Aggregate',     2)    # aggrega vincoli simili 2
    model.setParam('PreMIQCPForm', 1)     # per una riformulazione più efficiente dei coni quadrati prima di entrare in barrier

    
    # ===== Looser tolerances =====
    model.setParam('BarQCPConvTol', 1e-6)  # barrier conv tol
    model.setParam('FeasibilityTol',5e-6)  # toll. fattibilità
    model.setParam('OptimalityTol', 1e-6)  # toll. ottimalità
    model.setParam("BarConvTol", 1e-5) #new with L fixed
    
    # ===== Gap =====
    model.setParam('MIPGap',        1e-4)  # gap relativo
    
    # ===== MIP Strategies =====
    model.setParam('MIPFocus',      1)     # favorisce soluzioni iniziali  1
    model.setParam('Heuristics',    0.5)   # euristiche ridotte   0.6
    model.setParam('Cuts',          1)     # tagli disabilitati   2
    model.setParam('VarBranch',     2)     # branching rule leggera 1
    model.setParam('ConcurrentMIP', 1)     # prova 2 configurazioni in parallelo 1
    
    # ===== Increased numerical precision =====
    model.setParam('NumericFocus', 2)     # 0=default, 1=più veloce, 2=più preciso
    
    
    # ===== Parallelism =====
    model.setParam('Threads',       8) 
    
    # Optimize
    model.optimize()
    
    if model.Status == GRB.OPTIMAL:
        print("\nSolution quality:")
        model.printQuality()
    
    if model.Status != GRB.OPTIMAL:
        raise ValueError(f"Model status {model.Status}: infeasible or unbounded")


    tickers = list(new_data_stocks_interval_1.keys())

    x_selected = [j for j in range(n) if x_tilde[j].X > 1e-4]  
    selected_tickers = [tickers[j] for j in x_selected]

    weights = {
        tickers[i]: x_tilde[i].X
        for i in x_selected
    }
    
    # Create DataFrame of weights
    weights_df = pd.DataFrame(weights.items(), columns=["Stock", "Weight"])

    # Optional sorting by descending weight
    weights_df = weights_df.sort_values(by='Weight', ascending=False).reset_index(drop=True)

    # Compute the norm of the difference
    differences = [weights[stock] - w0_1.get(stock, 0) for stock in weights]
    norm_diff = np.linalg.norm(differences)

    # Optimal objective function value
    objective_value = model.objVal
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Model status code: {model.Status}")

    return objective_value, weights_df, selected_tickers, norm_diff, total_time

