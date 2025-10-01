import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_metrics(d_values, results, n_value):
    """
    Generates and saves plots for ARI and NMI metrics.
    """
    stats = {}
    for key, data in results.items():
        if key.startswith(('ari', 'nmi')):
            stats[f'mean_{key}'] = np.mean(data, axis=0)
            stats[f'std_{key}'] = np.std(data, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Clustering Performance (n = {n_value} samples per component)', fontsize=16)

    # --- ARI Plot ---
    ax = axes[0]
    # GMM-Grad
    ax.plot(d_values, stats['mean_ari_grad'], label="GMM-grad", color="royalblue")
    ax.fill_between(d_values, stats['mean_ari_grad'] - stats['std_ari_grad'], stats['mean_ari_grad'] + stats['std_ari_grad'], alpha=0.5, color="lightsteelblue")
    # GMM-GS
    ax.plot(d_values, stats['mean_ari_gs'], label="GMM-GS", color="orange")
    ax.fill_between(d_values, stats['mean_ari_gs'] - stats['std_ari_gs'], stats['mean_ari_gs'] + stats['std_ari_gs'], alpha=0.5, color="peachpuff")
    # Standard GMM
    ax.plot(d_values, stats['mean_ari_gmm'], label="GMM", color="gray")
    ax.fill_between(d_values, stats['mean_ari_gmm'] - stats['std_ari_gmm'], stats['mean_ari_gmm'] + stats['std_ari_gmm'], alpha=0.1, color="gray")
    
    ax.set_title("Adjusted Rand Index (ARI)")
    ax.set_xlabel("Number of features (d)")
    ax.set_ylabel("ARI Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # --- NMI Plot ---
    ax = axes[1]
    # GMM-GRAD
    ax.plot(d_values, stats['mean_nmi_grad'], label="GMM-Grad", color="royalblue")
    ax.fill_between(d_values, stats['mean_nmi_grad'] - stats['std_nmi_grad'], stats['mean_nmi_grad'] + stats['std_nmi_grad'], alpha=0.5, color="lightsteelblue")
    # GMM-GS
    ax.plot(d_values, stats['mean_nmi_gs'], label="GMM-GS", color="orange")
    ax.fill_between(d_values, stats['mean_nmi_gs'] - stats['std_nmi_gs'], stats['mean_nmi_gs'] + stats['std_nmi_gs'], alpha=0.5, color="peachpuff")
    # Standard GMM
    ax.plot(d_values, stats['mean_nmi_gmm'], label="GMM", color="gray")
    ax.fill_between(d_values, stats['mean_nmi_gmm'] - stats['std_nmi_gmm'], stats['mean_nmi_gmm'] + stats['std_nmi_gmm'], alpha=0.1, color="gray")

    ax.set_title("Normalized Mutual Information (NMI)")
    ax.set_xlabel("Number of features (d)")
    ax.set_ylabel("NMI Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = "simulation_plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"plot_results_n{n_value}.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")
    plt.close()

def run_plotting(args):
    """Main function to load results and generate plots."""
    try:
        data = np.load(args.input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_file}'")
        sys.exit(1)

    d_values = data['d_values']
    n_value = data['n_value']

    results = {key: data[key] for key in data if key not in ['d_values', 'n_value']}
    
    plot_metrics(d_values, results, n_value)
