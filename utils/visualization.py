# utils/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def plot_auroc_comparison(results_dict):
    """
    Upgraded academic-style layer-wise AUROC plot for baselines and probes.
    """
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(12, 8))

    max_layer = 0
    best_our_auc, best_our_layer = 0, 0
    
    styles = {
        # 1. Ours & Reference Lines
        'Ours (FF Probe + PeerNorm)':   {'color': '#E74C3C', 'marker': 'o', 'lw': 3.5, 'ls': '-'},   # Crimson Red (Highlight)
        'Ablation (FF No PeerNorm)':    {'color': '#E67E22', 'marker': 'v', 'lw': 2.0, 'ls': '--'},  # Orange
        'Baseline (Probability)':       {'color': '#7F8C8D', 'marker': '',  'lw': 2.5, 'ls': '--'},  # Flat Gray
        
        # 2. Advanced Baselines - Deep DL (Dark/Navy Blues)
        'Baseline (SAPLMA)':            {'color': '#0B2447', 'marker': 'p', 'lw': 2.0, 'ls': '-'},   # Navy Blue
        'Baseline (NL-CCS)':            {'color': '#154360', 'marker': 'P', 'lw': 2.0, 'ls': '--'},  # Deep Ocean Blue
        'Baseline (ConceptBottleneck)': {'color': '#1A5276', 'marker': '*', 'lw': 2.0, 'ls': '-.'},  # Dark Blue
        
        # 3. Advanced Baselines - Statistical (Medium/Bright Blues)
        'Baseline (RepE)':              {'color': '#2471A3', 'marker': 'v', 'lw': 2.0, 'ls': '-'},   # Strong Blue
        'Baseline (Mahalanobis)':       {'color': '#2980B9', 'marker': '^', 'lw': 2.0, 'ls': '--'},  # Medium Cobalt
        'Baseline (LDA)':               {'color': '#3498DB', 'marker': '<', 'lw': 2.0, 'ls': '-.'},  # Bright Azure
        'Baseline (KNN)':               {'color': '#5DADE2', 'marker': '>', 'lw': 2.0, 'ls': ':'},   # Sky Blue
        
        # 4. Basic Baselines (Light/Soft Blues)
        'Baseline (MLP)':               {'color': '#4A90E2', 'marker': 'D', 'lw': 2.0, 'ls': '-'},   # Dodger Blue (Strongest basic)
        'Baseline (CCS)':               {'color': '#7FB3D5', 'marker': 'X', 'lw': 2.0, 'ls': '--'},  # Soft Blue
        'Baseline (LR)':                {'color': '#A9CCE3', 'marker': 's', 'lw': 2.0, 'ls': '-.'},  # Light Blue
        'Baseline (Mass-Mean)':         {'color': '#D4E6F1', 'marker': 'd', 'lw': 2.0, 'ls': ':'}    # Very Light Blue
    }
    for label, aurocs in results_dict.items():
        # Handle flat lines (like Probability / Entropy)
        # Convert numpy array to list for set() check if needed
        aurocs_list = aurocs.tolist() if isinstance(aurocs, np.ndarray) else aurocs
        
        if isinstance(aurocs, float) or len(set(aurocs_list)) == 1:
            val = aurocs if isinstance(aurocs, float) else aurocs_list[0]
            style = styles.get(label, {'color': 'gray', 'ls': '--', 'lw': 2})
            ax.axhline(val, label=f"{label} ({val:.3f})", color=style['color'], 
                       linestyle=style['ls'], linewidth=style['lw'], alpha=0.9)
            continue

        layers = np.arange(len(aurocs))
        max_layer = max(max_layer, len(layers) - 1)
        style = styles.get(label, {'color': 'gray', 'marker': '.', 'lw': 1.5, 'ls': '-'})

        ax.plot(layers, aurocs, label=label, color=style['color'], 
                linewidth=style['lw'], linestyle=style['ls'], 
                marker=style['marker'], markersize=8, alpha=0.85)

        if label == 'Ours (FF Probe + PeerNorm)':
            best_our_auc = float(np.max(aurocs))
            best_our_layer = int(np.argmax(aurocs))

    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, label='Random Chance (0.500)')
    
    ax.annotate(f'Peak Detection\nLayer {best_our_layer}\nAUC: {best_our_auc:.3f}',
                xy=(best_our_layer, best_our_auc),
                xytext=(best_our_layer - 4, best_our_auc - 0.12),
                arrowprops=dict(facecolor='#E74C3C', shrink=0.05, width=1.5, headwidth=7),
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#E74C3C", lw=1.5),
                fontsize=12, fontweight='bold', color='#E74C3C')

    ax.set_title("Layer-wise Hallucination Detection: FFprobe vs. Baselines", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Layer Depth", fontsize=15, fontweight='bold')
    ax.set_ylabel("AUROC Score", fontsize=15, fontweight='bold')
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(-1, max_layer + 1)
    
    ax.legend(loc='upper left', frameon=True, fontsize=11, fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'auroc_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def generate_academic_baseline_table(results_dict):
    """
    Generates an academic-style Benchmark Table extracting the peak AUROC for each method.
    """
    data = []
    for label, aurocs in results_dict.items():
        aurocs_list = aurocs.tolist() if isinstance(aurocs, np.ndarray) else aurocs
        
        if isinstance(aurocs, float):
            score = aurocs
            layer_used = "N/A (Output)"
        elif len(set(aurocs_list)) == 1:
            score = aurocs_list[0]
            layer_used = "N/A (Output)"
        else:
            score = float(np.max(aurocs))
            layer_used = int(np.argmax(aurocs))

        data.append({
            "Method": label,
            "Best Layer": layer_used,
            "SimpleQA (ID) AUROC": score
        })

    df = pd.DataFrame(data)
    df.sort_values(by="SimpleQA (ID) AUROC", ascending=False, inplace=True)
    df["SimpleQA (ID) AUROC"] = df["SimpleQA (ID) AUROC"].apply(lambda x: f"{x:.4f}")
    df.reset_index(drop=True, inplace=True)

    print("\n" + "="*65)
    print("🔬 Benchmark Comparison Table (Evaluated at Peak Layer)")
    print("="*65)
    print(df.to_markdown(index=False, tablefmt="github"))
    print("="*65 + "\n")

    latex_str = df.to_latex(index=False, caption="Baseline comparison of hallucination detection AUROC.", label="tab:baselines")
    with open(os.path.join(RESULTS_DIR, 'benchmark_table.txt'), 'w') as f:
        f.write(df.to_markdown(index=False, tablefmt="github"))
        f.write('\n\n--- LaTeX ---\n\n')
        f.write(latex_str)

    return df

def plot_peernorm_ablation(full_aurocs, no_peernorm_aurocs):
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = np.arange(len(full_aurocs))

    ax.plot(layers, full_aurocs, label='FFprobe (Full: w/ PeerNorm)',
            color='#E74C3C', linewidth=3.0, linestyle='-', marker='o', markersize=7, alpha=0.9)
    ax.plot(layers, no_peernorm_aurocs, label='Ablation (w/o PeerNorm)',
            color='#E67E22', linewidth=2.5, linestyle='--', marker='v', markersize=7, alpha=0.9)
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1.5, label='Random Chance')

    ax.set_title("Ablation Study: Impact of PeerNorm on Layer-wise Stability", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Layer Depth", fontsize=14, fontweight='bold')
    ax.set_ylabel("AUROC Score", fontsize=14, fontweight='bold')

    y_min = min(min(full_aurocs), min(no_peernorm_aurocs)) - 0.05
    ax.set_ylim(max(0.4, y_min), 1.05)
    ax.set_xlim(-1, len(layers))
    ax.legend(loc='lower right', frameon=True, fontsize=12, fancybox=True, shadow=True)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'peernorm_ablation.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_origin_tracing(sqa_aucs, lqa_aucs):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    layers = list(range(len(sqa_aucs)))
    plt.plot(layers, sqa_aucs, marker='o', color='#E74C3C', linewidth=2.5, markersize=8,
             label='SimpleQA (Factual Memory)')
    plt.plot(layers, lqa_aucs, marker='^', color='#8E44AD', linewidth=2.5, markersize=8,
             label='LogiQA (Logical Routing)')

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.title('Layer-wise Origin Tracing: Facts vs. Logic in Llama-3', fontsize=16, pad=15)
    plt.xlabel('Transformer Layer Depth', fontsize=14)
    plt.ylabel('FF Probe AUROC', fontsize=14)
    plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    plt.ylim(0.4, 1.0)
    plt.xticks(range(0, len(layers), 2))

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'origin_tracing.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_snr_distribution(pos_scores, neg_scores, layer_idx, learned_threshold=None):
    """
    Plots the energy (Goodness) distribution for Truth vs. Hallucination 
    and calculates the Signal-to-Noise Ratio (SNR).
    """
    mean_pos, std_pos = np.mean(pos_scores), np.std(pos_scores)
    mean_neg, std_neg = np.mean(neg_scores), np.std(neg_scores)
    
    # SNR Calculation
    dist = abs(mean_pos - mean_neg)
    std_avg = (std_pos + std_neg) / 2
    snr = dist / (std_avg + 1e-6)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.kdeplot(pos_scores, fill=True, color='#2ecc71', label='Truth (Real)', alpha=0.4, linewidth=2)
    sns.kdeplot(neg_scores, fill=True, color='#e74c3c', label='Hallucination (Fake)', alpha=0.4, linewidth=2)
    
    plt.axvline(mean_pos, color='#27ae60', linestyle=':', linewidth=1.5, label=f'Mean Truth ({mean_pos:.2f})')
    plt.axvline(mean_neg, color='#c0392b', linestyle=':', linewidth=1.5, label=f'Mean Fake ({mean_neg:.2f})')
    
    if learned_threshold is not None:
        plt.axvline(learned_threshold, color='#2980b9', linestyle='-', linewidth=2.5,
                    label=f'Learned Threshold ({learned_threshold:.2f})')
    
    plt.title(f"FF Probe Energy Distribution (Layer {layer_idx})\nCalculated SNR: {snr:.4f}", fontsize=14)
    plt.xlabel("Goodness Score (Energy)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    all_scores = np.concatenate([pos_scores, neg_scores])
    plt.xlim(all_scores.min()*0.9, all_scores.max()*1.1)
    
    plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'snr_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_energy_landscape(g_sqa_pos, g_sqa_neg, g_lqa_pos, g_lqa_neg, target_layer):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.kdeplot(g_sqa_pos, fill=True, color='#2ECC71', alpha=0.5, linewidth=2, label='True Fact (Pos)', ax=axes[0])
    sns.kdeplot(g_sqa_neg, fill=True, color='#E74C3C', alpha=0.5, linewidth=2, label='Hallucination (Neg)', ax=axes[0])
    axes[0].set_title(f'Source Domain (SimpleQA) - Layer {target_layer}')
    axes[0].legend(loc='upper right', frameon=True)

    sns.kdeplot(g_lqa_pos, fill=True, color='#2ECC71', alpha=0.5, linewidth=2, label='Correct Logic (Pos)', ax=axes[1])
    sns.kdeplot(g_lqa_neg, fill=True, color='#E74C3C', alpha=0.5, linewidth=2, label='Logical Error (Neg)', ax=axes[1])
    axes[1].set_title(f'Target Domain (LogiQA) - Layer {target_layer}')
    axes[1].legend(loc='upper right', frameon=True)

    plt.suptitle('Cross-Domain Goodness Landscape: Feature Reversal Phenomenon', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'energy_landscape.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_logiqa_landscape(g_lqa_pos, g_lqa_neg, target_layer):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.kdeplot(g_lqa_pos, fill=True, color='#2ECC71', alpha=0.6, linewidth=2, label='Correct Logic (Pos)')
    sns.kdeplot(g_lqa_neg, fill=True, color='#E74C3C', alpha=0.6, linewidth=2, label='Logical Error (Neg)')

    plt.title(f'Target Domain (LogiQA) - Dedicated Probe at Layer {target_layer}', fontsize=15, pad=15)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'logiqa_landscape.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_neuron_attribution(sqa_weights, lqa_weights, sqa_top_indices, overlap_ratio, top_k):
    sqa_vals_in_sqa = sqa_weights[sqa_top_indices].numpy()
    sqa_vals_in_lqa = lqa_weights[sqa_top_indices].numpy()

    plt.figure(figsize=(14, 6))
    x = np.arange(top_k)
    width = 0.4

    plt.bar(x - width/2, sqa_vals_in_sqa, width, label='Importance in Fact Probe', color='#E74C3C')
    plt.bar(x + width/2, sqa_vals_in_lqa, width, label='Importance in Logic Probe', color='#8E44AD', alpha=0.7)

    plt.title(f'Cognitive Orthogonality: Top {top_k} Factual Neurons are Ignored by Logic Probe\n'
              f'(Overlap: {overlap_ratio*100:.1f}%)', fontsize=16, pad=15)
    plt.xlabel('Top Factual Neuron Indices (Sorted by Importance)', fontsize=14)
    plt.ylabel('Mean Absolute Weight Magnitude', fontsize=14)
    plt.legend(loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'neuron_attribution.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_topk_overlap_sweep(overlap_ratios, random_chances, k_list):
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.figure(figsize=(10, 6))

    plt.plot(k_list, overlap_ratios, marker='o', color='#8E44AD', linewidth=3, markersize=8,
             label='Empirical Neuron Overlap')
    plt.plot(k_list, random_chances, linestyle='--', color='gray', linewidth=2,
             label='Theoretical Random Chance')

    plt.title('Cognitive Separation: Neuron Overlap Across Top-K Sweep', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Top-K Neurons Selected', fontsize=14, fontweight='bold')
    plt.ylabel('Overlap Ratio', fontsize=14, fontweight='bold')

    max_y = max(max(overlap_ratios), max(random_chances)) + 0.05
    plt.ylim(-0.01, max(0.2, max_y))

    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'topk_overlap_sweep.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_goodness_trajectory(generated_tokens, g_scores, target_layer, save_name=None):
    plt.figure(figsize=(14, 6))
    x_pos = np.arange(len(generated_tokens))

    norm = plt.Normalize(min(g_scores), max(g_scores))
    colors = plt.cm.RdYlGn(norm(g_scores))

    plt.plot(x_pos, g_scores, color='gray', linestyle='-', alpha=0.5, zorder=1)
    plt.scatter(x_pos, g_scores, c=colors, s=100, edgecolor='black', zorder=2)

    plt.xticks(x_pos, [t.replace('\n', '\\n') for t in generated_tokens], rotation=60, ha='right', fontsize=9)
    plt.title(f'Real-time Hallucination Trajectory (Layer {target_layer})', fontsize=16, pad=15)
    plt.ylabel('FF Goodness Score', fontsize=14)
    plt.xlabel('Generated Tokens', fontsize=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    fname = save_name or 'goodness_trajectory'
    plt.savefig(os.path.join(RESULTS_DIR, f'{fname}.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_causal_ablation(g_original, g_ablated, g_rand, top_k, target_layer):
    labels = ['Original\n(Hallucination)', f'Top-{top_k} Ablated\n(Targeted)', f'Random-{top_k} Ablated\n(Control)']
    means = [g_original.mean(), g_ablated.mean(), g_rand.mean()]
    stds  = [g_original.std(),  g_ablated.std(),  g_rand.std()]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, means, yerr=stds, capsize=10,
                   color=['#E74C3C', '#2ECC71', '#95A5A6'], alpha=0.85, edgecolor='black')

    plt.ylabel('FF Goodness Score', fontsize=12)
    plt.title(f'Causal Ablation via Goodness Gradient (Layer {target_layer})', fontsize=14)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.3f}',
                 ha='center', va='bottom', fontsize=11)

    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'causal_ablation.png'), dpi=300, bbox_inches='tight')
    plt.show()

