# utils/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    ax.legend(loc='lower left', frameon=True, fontsize=11, fancybox=True, shadow=True, ncol=2)

    fig.tight_layout()
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
    """
    # Generate LaTeX code for direct pasting into Overleaf
    print("💡 LaTeX Code for your paper:")
    print(df.to_latex(index=False, caption="Baseline comparison of hallucination detection AUROC.", label="tab:baselines"))
    """
    
    return df

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
        plt.text(learned_threshold, plt.ylim()[1]*0.85, " Predicted Truth →", 
                 color='#2980b9', fontweight='bold', va='center')
    
    plt.title(f"FF Probe Energy Distribution (Layer {layer_idx})\nCalculated SNR: {snr:.4f}", fontsize=14)
    plt.xlabel("Goodness Score (Energy)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    
    all_scores = np.concatenate([pos_scores, neg_scores])
    plt.xlim(all_scores.min()*0.9, all_scores.max()*1.1)
    
    plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()
    plt.show()
