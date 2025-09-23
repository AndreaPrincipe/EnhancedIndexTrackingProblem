import matplotlib.pyplot as plt
import io
from PIL import Image
import os

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 16,
})


def plot_norm_differences(q_values, results_model_1, results_model_2, results_model_3, save_figures, results_path):
    """
    Function to plot a graph of the norm differences for each model as q varies.
    """
    
    plt.figure(figsize=(10, 6))

    # Plot lines for each model
    plt.plot(q_values, [results_model_1[q][3] for q in q_values], label="Model 1", marker='o')
    plt.plot(q_values, [results_model_2[q][3] for q in q_values], label="Model 2", marker='s')
    plt.plot(q_values, [results_model_3[q][3] for q in q_values], label="Model 3", marker='^')

    # Add labels and title
    plt.xlabel('Portfolio size q')
    plt.ylabel('Norm of differences')
    plt.title('Norm trend for each model as q varies')

    # Show all q_values on the x-axis
    plt.xticks(q_values, labels=[str(q) for q in q_values])

    # Add legend and grid
    plt.legend(loc="lower right", bbox_to_anchor=(1, 0.08))
    plt.grid(True)

    # Save the figure
    if save_figures:
        plt.savefig(os.path.join(results_path, "norm_differences_q.png"))
        print("Graph saved as norm_differences_q")




def plot_objective_values(q_values, results_model_1, results_model_2, results_model_3, save_figures, results_path):
    """
    Function to create and save plots comparing the objective values of the models as a function of q.
    """

    plt.figure(figsize=(12, 8))
    plt.plot(q_values, [results_model_1[q][0] for q in q_values], label="Model 1", marker='o')
    plt.plot(q_values, [results_model_2[q][0] for q in q_values], label="Model 2", marker='s')
    plt.plot(q_values, [results_model_3[q][0] for q in q_values], label="Model 3", marker='^')
    
    plt.xlabel('Portfolio size q')
    plt.ylabel('Objective Value')
    plt.title('Objective Value Comparison as q Varies')
    plt.xticks(q_values)
    plt.legend()
    plt.grid(True)

    if save_figures:
        plt.savefig(os.path.join(results_path, "ObjValue_q.png"))
        

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))
    model_results = [results_model_1, results_model_2, results_model_3]
    model_titles = ['Model 1', 'Model 2', 'Model 3']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']
    colors = ['blue', 'green', 'red']

    for idx, ax in enumerate(axs.flat):
        ax.plot(q_values, [model_results[idx][q][0] for q in q_values], label=model_titles[idx],
                marker=markers[idx], linestyle=linestyles[idx], color=colors[idx])
        ax.set_title(model_titles[idx])
        ax.set_xlabel('Portfolio size q')
        ax.set_ylabel('Objective Value')
        ax.legend()
        ax.grid(True)

    fig.suptitle('Objective Value Comparison as q Varies', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_figures:
        plt.savefig(os.path.join(results_path, "ObjValue_q_multiple.png"))




def plot_portfolio_return_comparison(q_values, index_return_1, results_model_1, results_model_2, results_model_3, save_figures, results_path):
    """
    Function to display the portfolio return comparison plot as the portfolio size varies.
    """
    fig, ax = plt.subplots(figsize=(12, 8))  

    ax.plot(q_values, [index_return_1 for _ in q_values], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(q_values, [results_model_1[q][4] for q in q_values], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(q_values, [results_model_2[q][4] for q in q_values], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(q_values, [results_model_3[q][4] for q in q_values], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Return')
    ax.set_title('Portfolio Return Comparison as q Varies')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    if save_figures:
        fig.savefig(os.path.join(results_path, "Portfolio_return_q.png"))

    plt.tight_layout()
    
    return fig



def plot_portfolio_variance_comparison(results_model_1, results_model_2, results_model_3, index_variance, q_values, save_figures, results_path):
    """
    Function to display the portfolio variance comparison plot as the portfolio size varies.
    """
    
    results_models = [results_model_1, results_model_2, results_model_3]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(q_values, [index_variance for _ in q_values], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(q_values, [results_models[0][q][5] for q in q_values], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(q_values, [results_models[1][q][5] for q in q_values], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(q_values, [results_models[2][q][5] for q in q_values], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Variance')
    ax.set_title('Portfolio Variance Comparison as q Varies')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.05))
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    if save_figures:
        save_path = os.path.join(results_path, "Portfolio_variance_q.png")
        fig.savefig(save_path)
        print(f"Plot saved as {save_path}")
    
    plt.tight_layout()
    
    return fig


def plot_sharpe_ratios_comparison(results_model_1, results_model_2, results_model_3, SR_index, q_values, save_figures, results_path):
    """
    Function to display the portfolio sharpe ratio comparison plot as the portfolio size varies.
    """
    
    results_models = [results_model_1, results_model_2, results_model_3]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(q_values, [SR_index for _ in q_values], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(q_values, [results_models[0][q][6] for q in q_values], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(q_values, [results_models[1][q][6] for q in q_values], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(q_values, [results_models[2][q][6] for q in q_values], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Sharpe Ratio')
    ax.set_title('Portfolio Sharpe Ratio Comparison as q Varies')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.13))
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    if save_figures:
        save_path = os.path.join(results_path, "Portfolio_sharpe_ratios_q.png")
        fig.savefig(save_path)
        print(f"Plot saved as {save_path}")

    plt.tight_layout()
    
    return fig


def figures_merge(fig1, fig2, save_figure, results_path, file_name):
    def figure_to_image(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='PNG')
        buf.seek(0)
        return Image.open(buf)

    img1 = figure_to_image(fig1)
    img2 = figure_to_image(fig2)
    
    h_min = min(img1.height, img2.height)
    img1 = img1.resize((img1.width, h_min))
    img2 = img2.resize((img2.width, h_min))
    new_width = img1.width + img2.width
    final_figure = Image.new('RGB', (new_width, h_min))
    final_figure.paste(img1, (0, 0))
    final_figure.paste(img2, (img1.width, 0))

    if save_figure:
        save_path = os.path.join(results_path, file_name)
        final_figure.save(save_path)
        print(f"Figure saved as {save_path}")
    
    return final_figure


def plot_tracking_ratio(tracking_ratio_model_1, tracking_ratio_model_2, tracking_ratio_model_3, q_values, save_figures, results_path):
    """
    Function to display the portfolio tracking ratio as the portfolio size varies.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(q_values, [1 for _ in q_values], label="Ideal Tracking Ratio", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot(tracking_ratio_model_1["q"], tracking_ratio_model_1["tracking_ratio"], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(tracking_ratio_model_2["q"], tracking_ratio_model_2["tracking_ratio"], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(tracking_ratio_model_3["q"], tracking_ratio_model_3["tracking_ratio"], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Tracking Ratio')
    ax.set_title('Comparison of Portfolio Tracking Ratios with Varying q')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    if save_figures:
        save_path = os.path.join(results_path, "Tracking_Ratios.png")
        fig.savefig(save_path)
        print(f"Plot saved as {save_path}")
    
    plt.tight_layout()



def plot_tracking_error(tracking_error_model_1, tracking_error_model_2, tracking_error_model_3, q_values, save_figures, results_path):
    """
    Function to display the portfolio tracking error plot as the portfolio size varies.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(tracking_error_model_1["q"], tracking_error_model_1["tracking_error"], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot(tracking_error_model_2["q"], tracking_error_model_2["tracking_error"], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
    ax.plot(tracking_error_model_3["q"], tracking_error_model_3["tracking_error"], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)

    ax.set_xlabel('Portfolio size q')
    ax.set_ylabel('Portfolio Tracking Error')
    ax.set_title('Comparison of Portfolio Tracking Error with Varying q')
    ax.set_xticks(q_values)
    ax.set_xticklabels([str(q) for q in q_values])
    ax.legend()
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)

    if save_figures:
        save_path = os.path.join(results_path, "Tracking_Error.png")
        fig.savefig(save_path)
        print(f"Plot saved as {save_path}")
    
    plt.tight_layout()
    


def plot_portfolio_return_rolling_windows(intervals, index_return_var, q_values_roll, results_model_1_roll, results_model_2_roll, results_model_3_roll, save_figures, results_path):
    """
    Plots portfolio returns over rolling windows, comparing them to the reference index for different q values.
    """
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))  
        ax.plot([interval[1] for interval in intervals], index_return_var['index_return'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][q][4] for interval in intervals], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][q][4] for interval in intervals], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_3_roll[interval][q][4] for interval in intervals], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Return')
        ax.set_title(f"Ex Post Portfolio Return q={q}")
        ax.set_xticks([interval[1] for interval in intervals])
        ax.set_xticklabels([str(interval[1]) for interval in intervals], rotation=30, ha="center")
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig(os.path.join(results_path, "Ex_Post_Portfolio_Return.png"))
            print("Ex_Post_Portfolio_Return.png")
            
        plt.tight_layout()
        


def plot_portfolio_variance_rolling_windows(intervals, index_return_var, q_values_roll, results_model_1_roll, results_model_2_roll, results_model_3_roll, save_figures, results_path):
    """
    Plots portfolio variance over rolling windows, comparing them to the reference index for different q values.
    """
    
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))  
        ax.plot([interval[1] for interval in intervals], index_return_var['index_variance'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][q][5] for interval in intervals], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][q][5] for interval in intervals], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_3_roll[interval][q][5] for interval in intervals], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Variance')
        ax.set_title(f"Ex Post Portfolio Variance q={q}")
        ax.set_xticks([interval[1] for interval in intervals])
        ax.set_xticklabels([str(interval[1]) for interval in intervals], rotation=30, ha="center")
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig(os.path.join(results_path, "Ex_Post_Portfolio_Variance.png"))
            print("Ex_Post_Portfolio_Variance.png")
            
        plt.tight_layout()
        


def plot_portfolio_sharpe_ratios_rolling_windows(intervals, index_return_var, q_values_roll, results_model_1_roll, results_model_2_roll, results_model_3_roll, save_figures, results_path):
    """
    Plots portfolio sharpe ratio over rolling windows, comparing them to the reference index for different q values.
    """
    
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))  
        ax.plot([interval[1] for interval in intervals], index_return_var['SR_index'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][q][6] for interval in intervals], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][q][6] for interval in intervals], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals], [results_model_3_roll[interval][q][6] for interval in intervals], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio sharpe ratios')
        ax.set_title(f"Ex Post Portfolio sharpe ratios q={q}")
        ax.set_xticks([interval[1] for interval in intervals])
        ax.set_xticklabels([str(interval[1]) for interval in intervals], rotation=30, ha="center")
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig(os.path.join(results_path, "Ex_Post_Portfolio_sharpe_ratios.png"))
            print("Ex_Post_Portfolio_sharpe_ratios.png")
            
        plt.tight_layout()
        


def plot_tracking_ratio_roll_out(q_values_roll, intervals_out, tracking_ratio_dict_1, tracking_ratio_dict_2, tracking_ratio_dict_3, save_figures, results_path):
    """
    Plots portfolio tracking ratio over rolling windows for different q values.
    """
    
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot([interval[1] for interval in intervals_out], [1 for _ in intervals_out], label="Ideal Tracking Ratio", marker='*', linestyle='-', color='blue', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_1[interval].loc[tracking_ratio_dict_1[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_2[interval].loc[tracking_ratio_dict_2[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_3[interval].loc[tracking_ratio_dict_3[interval]['q'] == q, 'tracking_ratio'].iloc[0] for interval in intervals_out], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Tracking Ratio')
        ax.set_title(f"Comparison of Portfolio Tracking Ratios with Intervals q={q}")
        ax.set_xticks([interval[1] for interval in intervals_out])
        ax.set_xticklabels([str(interval[1]) for interval in intervals_out], rotation=30, ha="center")
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig(os.path.join(results_path, "Tracking_Ratios_Roll_out.png"))
            print("Plot saved as Tracking_Ratios_Roll_out.png")
            
        plt.tight_layout()
        


def plot_tracking_error_roll_out(q_values_roll, intervals_out, tracking_error_dict_1, tracking_error_dict_2, tracking_error_dict_3, save_figures, results_path):
    """
    Plots portfolio tracking error over rolling windows for different q values.
    """
    
    for q in q_values_roll:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_1[interval][q] for interval in intervals_out], label="Model 1", marker='o', linestyle='-', color='orange', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_2[interval][q] for interval in intervals_out], label="Model 2", marker='s', linestyle='-', color='green', markersize=8)
        ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_3[interval][q] for interval in intervals_out], label="Model 3", marker='^', linestyle='-', color='red', markersize=8)
        ax.set_xlabel('Intervals')
        ax.set_ylabel('Portfolio Tracking Error')
        ax.set_title(f"Comparison of Portfolio Tracking Error with Intervals q={q}")
        ax.set_xticks([interval[1] for interval in intervals_out])
        ax.set_xticklabels([str(interval[1]) for interval in intervals_out], rotation=30, ha="center")
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
        
        if save_figures:
            fig.savefig(os.path.join(results_path, "Tracking_Error_Roll_out.png"))
            print("Plot saved as Tracking_Error_Roll_Out.png")
            
        plt.tight_layout()
        

def plot_portfolio_return_out_mixture(intervals, index_return_var, results_model_1_roll, results_model_2_roll, save_figures, results_path):
    """
    Plot out-of-sample portfolio returns over time, comparing benchmark index and two models (EIT and DEIT).
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))  
    ax.plot([interval[1] for interval in intervals], index_return_var['index_return'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][4] for interval in intervals], label="EIT", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][4] for interval in intervals], label="DEIT", marker='s', linestyle='-', color='green', markersize=8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Portfolio Return')
    ax.set_title("Out-of-Sample Portfolio Returns")
    ticks = [interval[1] for interval in intervals]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(date) if i % 2 == 0 else "" for i, date in enumerate(ticks)])
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.02))

    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    if save_figures:
        fig.savefig(os.path.join(results_path, "Portfolio_Returns.png"))
        print("Portfolio_Returns.png")
    
    plt.tight_layout()


def plot_portfolio_variance_out_mixture(intervals, index_return_var, results_model_1_roll, results_model_2_roll, save_figures, results_path):
    """
    Plot out-of-sample portfolio variance over time, comparing benchmark index and two models (EIT and DEIT).
    """
    fig, ax = plt.subplots(figsize=(12, 8))  
    ax.plot([interval[1] for interval in intervals], index_return_var['index_variance'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][5] for interval in intervals], label="EIT", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][5] for interval in intervals], label="DEIT", marker='s', linestyle='-', color='green', markersize=8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Portfolio Variance')
    ax.set_title("Out-of-Sample Portfolio Variance")
    ticks = [interval[1] for interval in intervals]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(date) if i % 2 == 0 else "" for i, date in enumerate(ticks)])
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    if save_figures:
        fig.savefig(os.path.join(results_path, "Portfolio_Variance.png"))
        print("Portfolio_Variance.png")
        
    plt.tight_layout()


def plot_portfolio_sharpe_ratios_out_mixture(intervals, index_return_var, results_model_1_roll, results_model_2_roll, save_figures, results_path):
    """
    Plot out-of-sample portfolio sharpe ratio over time, comparing benchmark index and two models (EIT and DEIT).
    """
    fig, ax = plt.subplots(figsize=(12, 8))  
    ax.plot([interval[1] for interval in intervals], index_return_var['SR_index'], label="S&P 500 index", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot([interval[1] for interval in intervals], [results_model_1_roll[interval][6] for interval in intervals], label="EIT", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot([interval[1] for interval in intervals], [results_model_2_roll[interval][6] for interval in intervals], label="DEIT", marker='s', linestyle='-', color='green', markersize=8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Portfolio Sharpe ratios')
    ax.set_title("Out-of-Sample Portfolio Sharpe ratios")
    ticks = [interval[1] for interval in intervals]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(date) if i % 2 == 0 else "" for i, date in enumerate(ticks)])
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.02))
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    if save_figures:
        fig.savefig(os.path.join(results_path, "Portfolio_sharpe_ratios.png"))
        print("Portfolio_sharpe_ratios.png")
        
    plt.tight_layout()


def plot_tracking_ratio_out_mixture(intervals_out, tracking_ratio_dict_1, tracking_ratio_dict_2, save_figures, results_path):
    """
    Plot out-of-sample portfolio tracking ratio over time, comparing two models (EIT and DEIT).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([interval[1] for interval in intervals_out], [1 for _ in intervals_out], label="Ideal Tracking Ratio", marker='*', linestyle='-', color='blue', markersize=8)
    ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_1[interval] for interval in intervals_out], label="EIT", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot([interval[1] for interval in intervals_out], [tracking_ratio_dict_2[interval] for interval in intervals_out], label="DEIT", marker='s', linestyle='-', color='green', markersize=8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Portfolio Tracking Ratio')
    ax.set_title("Out-of-Sample Portfolio Tracking Ratios")
    ticks = [interval[1] for interval in intervals_out]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(date) if i % 2 == 0 else "" for i, date in enumerate(ticks)])
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    if save_figures:
        fig.savefig(os.path.join(results_path, "Tracking_Ratios.png"))
        print("Plot saved as Tracking_Ratios.png")
        
    plt.tight_layout()


def plot_tracking_error_out_mixture(intervals_out, tracking_error_dict_1, tracking_error_dict_2, save_figures, results_path):
    """
    Plot out-of-sample portfolio tracking error over time, comparing two models (EIT and DEIT).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_1[interval] for interval in intervals_out], label="EIT", marker='o', linestyle='-', color='orange', markersize=8)
    ax.plot([interval[1] for interval in intervals_out], [tracking_error_dict_2[interval] for interval in intervals_out], label="DEIT", marker='s', linestyle='-', color='green', markersize=8)
    ax.set_xlabel('Period')
    ax.set_ylabel('Portfolio Tracking Error')
    ax.set_title("Out-of-Sample Portfolio Tracking Error")
    ticks = [interval[1] for interval in intervals_out]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(date) if i % 2 == 0 else "" for i, date in enumerate(ticks)])
    ax.legend(loc='best')
    ax.grid(True, which='both', linestyle='--', color='grey', alpha=0.7)
    
    if save_figures:
        fig.savefig(os.path.join(results_path, "Tracking_Error.png"))
        print("Plot saved as Tracking_Error.png")
        
    plt.tight_layout()
