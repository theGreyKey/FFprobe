# utils/__init__.py

# Expose data loading functions
from .data_loader import (
    prepare_simpleqa_data,
    prepare_logiqa_data,
)

# Expose baseline models and evaluation functions
from .baselines import (
    calculate_lr_auroc,
    calculate_mass_mean_auroc,
    calculate_mlp_auroc,
    calculate_ccs_auroc,
    calculate_prob_entropy_auroc,
    calculate_ff_no_peernorm_auroc,
    calculate_ff_no_zscore_auroc,
    StandardMLP,
    FFLayerProbeNoPeerNorm,
    FFLayerProbeNoZScore,
)

# Expose advanced baselines
from .advanced_baselines import (
    calculate_knn_auroc,
    calculate_mahalanobis_auroc,
    calculate_lda_auroc,
    calculate_repe_auroc,
    calculate_nlccs_auroc,
    calculate_saplma_auroc,
    calculate_concept_bottleneck_auroc,
)

# Expose evaluation utilities
from .evaluation import (
    compute_full_metrics,
    bootstrap_auroc_ci,
    bootstrap_metric_ci,
    mcnemar_test,
    delong_test,
    evaluate_probe_per_layer,
    compare_probes,
    print_evaluation_report,
    get_method_scores_at_layer,
)

# Expose visualization tools
from .visualization import (
    plot_auroc_comparison,
    plot_snr_distribution,
    generate_academic_baseline_table,
    plot_peernorm_ablation,
    plot_origin_tracing,
    plot_energy_landscape,
    plot_logiqa_landscape,
    plot_neuron_attribution,
    plot_topk_overlap_sweep,
    plot_goodness_trajectory,
    plot_causal_ablation,
)

__all__ = [
    # data_loader
    "prepare_simpleqa_data",
    "prepare_logiqa_data",
    # baselines — functions
    "calculate_lr_auroc",
    "calculate_mass_mean_auroc",
    "calculate_mlp_auroc",
    "calculate_ccs_auroc",
    "calculate_prob_entropy_auroc",
    "calculate_ff_no_peernorm_auroc",
    "calculate_ff_no_zscore_auroc",
    # baselines — models
    "StandardMLP",
    "FFLayerProbeNoPeerNorm",
    "FFLayerProbeNoZScore",
    # advanced baselines
    "calculate_knn_auroc",
    "calculate_mahalanobis_auroc",
    "calculate_lda_auroc",
    "calculate_repe_auroc",
    "calculate_nlccs_auroc",
    "calculate_saplma_auroc",
    "calculate_concept_bottleneck_auroc",
    # evaluation
    "compute_full_metrics",
    "bootstrap_auroc_ci",
    "bootstrap_metric_ci",
    "mcnemar_test",
    "delong_test",
    "evaluate_probe_per_layer",
    "compare_probes",
    "print_evaluation_report",
    "get_method_scores_at_layer",
    # visualization
    "plot_auroc_comparison",
    "plot_snr_distribution",
    "generate_academic_baseline_table",
    "plot_peernorm_ablation",
    "plot_origin_tracing",
    "plot_energy_landscape",
    "plot_logiqa_landscape",
    "plot_neuron_attribution",
    "plot_topk_overlap_sweep",
    "plot_goodness_trajectory",
    "plot_causal_ablation",
]
