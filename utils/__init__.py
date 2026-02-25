# utils/__init__.py

# Expose data loading functions
from .data_loader import (
    prepare_simpleqa_data,
    prepare_logiqa_data
)

# Expose baseline models and evaluation functions
from .baselines import (
    calculate_lr_auroc,
    calculate_mass_mean_auroc,
    StandardMLP,
    calculate_mlp_auroc,
    calculate_ccs_auroc,
    calculate_prob_entropy_auroc,
    FFLayerProbeNoPeerNorm,
    calculate_ff_no_peernorm_auroc,
    FFLayerProbeNoZScore,
    calculate_ff_no_zscore_auroc
)

# Expose visualization tools
from .visualization import (
    plot_auroc_comparison,
    plot_snr_distribution,
    generate_academic_baseline_table
)

__all__ = [
    "prepare_simpleqa_data",
    "prepare_logiqa_data",
    "calculate_lr_auroc",
    "calculate_mass_mean_auroc",
    "StandardMLP",
    "calculate_mlp_auroc",
    "calculate_ccs_auroc",
    "calculate_prob_entropy_auroc",
    "FFLayerProbeNoPeerNorm",
    "calculate_ff_no_peernorm_auroc",
    "FFLayerProbeNoZScore",
    "calculate_ff_no_zscore_auroc",
    "plot_auroc_comparison",
    "plot_snr_distribution",
    "generate_academic_baseline_table"
]