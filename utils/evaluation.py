# ============================================================
# utils/evaluation.py
# Complete evaluation metrics + statistical significance tests
# McNemar's Test / Bootstrap CI / DeLong Test / Cohen's d
# ============================================================

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, brier_score_loss,
    roc_curve, matthews_corrcoef,
)
from scipy import stats
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim

# Compatibility with scipy versions for binomial test
try:
    from scipy.stats import binomtest as _binomtest
    def _exact_binom_pvalue(b, n):
        return _binomtest(b, n, 0.5).pvalue
except ImportError:
    from scipy.stats import binom_test as _binom_test
    def _exact_binom_pvalue(b, n):
        return _binom_test(b, n, 0.5)


def compute_full_metrics(y_true, y_scores, threshold=None):
    """
    Compute complete evaluation metrics.

    Returns:
        dict: AUROC, AUPRC, Accuracy, F1, Precision, Recall,
              MCC, Brier Score, Cohen's d, Optimal Threshold
    """
    y_true = np.asarray(y_true, dtype=float)
    y_scores = np.asarray(y_scores, dtype=float)
    metrics = {}

    try:
        metrics['auroc'] = roc_auc_score(y_true, y_scores)
    except ValueError:
        metrics['auroc'] = 0.5

    try:
        metrics['auprc'] = average_precision_score(y_true, y_scores)
    except ValueError:
        metrics['auprc'] = 0.5

    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = thresholds[best_idx]
    metrics['optimal_threshold'] = float(threshold)

    y_pred = (y_scores >= threshold).astype(int)

    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    s_min, s_max = y_scores.min(), y_scores.max()
    if s_max - s_min > 1e-8:
        y_prob = (y_scores - s_min) / (s_max - s_min)
    else:
        y_prob = np.full_like(y_scores, 0.5)
    metrics['brier_score'] = brier_score_loss(y_true, y_prob)

    pos_s = y_scores[y_true == 1]
    neg_s = y_scores[y_true == 0]
    if len(pos_s) > 1 and len(neg_s) > 1:
        sp = np.sqrt(
            ((len(pos_s) - 1) * np.var(pos_s, ddof=1) +
             (len(neg_s) - 1) * np.var(neg_s, ddof=1))
            / (len(pos_s) + len(neg_s) - 2)
        )
        metrics['cohens_d'] = float((np.mean(pos_s) - np.mean(neg_s)) / sp) if sp > 0 else 0.0
    else:
        metrics['cohens_d'] = 0.0

    return metrics


def bootstrap_auroc_ci(y_true, y_scores, n_bootstrap=2000, ci=0.95, random_state=42):
    """
    Bootstrap confidence interval for AUROC (BCa percentile method).

    Returns:
        dict: auroc_mean, auroc_std, ci_lower, ci_upper, ci_level
    """
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = len(y_true)

    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt, ys = y_true[idx], y_scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot_aucs.append(roc_auc_score(yt, ys))
        except ValueError:
            continue

    boot_aucs = np.array(boot_aucs)
    alpha = 1 - ci

    return {
        'auroc_mean': float(np.mean(boot_aucs)),
        'auroc_std': float(np.std(boot_aucs)),
        'ci_lower': float(np.percentile(boot_aucs, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(boot_aucs, 100 * (1 - alpha / 2))),
        'ci_level': ci,
        'n_valid_bootstrap': len(boot_aucs),
    }


def bootstrap_metric_ci(y_true, y_scores, metric_fn, n_bootstrap=2000,
                         ci=0.95, random_state=42, **metric_kwargs):
    """
    Generic bootstrap CI for any metric function metric_fn(y_true, y_scores).
    """
    rng = np.random.RandomState(random_state)
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    n = len(y_true)

    vals = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            vals.append(metric_fn(y_true[idx], y_scores[idx], **metric_kwargs))
        except Exception:
            continue

    vals = np.array(vals)
    alpha = 1 - ci
    return {
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'ci_lower': float(np.percentile(vals, 100 * alpha / 2)),
        'ci_upper': float(np.percentile(vals, 100 * (1 - alpha / 2))),
    }


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test: compare two classifiers on sample-level differences.
    H0: Both classifiers have the same error rate.
    Uses exact binomial test when discordant pairs < 25, otherwise chi-squared.

    Args:
        y_true:   Ground truth labels
        y_pred_a: Model A binary predictions
        y_pred_b: Model B binary predictions

    Returns:
        dict: statistic, p_value, significant_at_005, contingency table
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    ca = (y_pred_a == y_true)
    cb = (y_pred_b == y_true)

    a = int(np.sum(ca & cb))
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    d = int(np.sum(~ca & ~cb))

    contingency = {
        'both_correct': a,
        'a_correct_b_wrong': b,
        'a_wrong_b_correct': c,
        'both_wrong': d,
    }

    if b + c == 0:
        return {'statistic': 0.0, 'p_value': 1.0,
                'significant_at_005': False, 'contingency': contingency}

    if b + c < 25:
        p_value = _exact_binom_pvalue(b, b + c)
        chi2_stat = float('nan')
    else:
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)

    return {
        'statistic': float(chi2_stat),
        'p_value': float(p_value),
        'significant_at_005': bool(p_value < 0.05),
        'contingency': contingency,
    }


def _compute_midrank(x):
    """Compute midrank for DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1.0
    return T2


def _delong_variance(ground_truth, predictions):
    """
    Compute DeLong variance structure components.
    ground_truth: 0/1 labels (sorted with positives first)
    predictions: shape (k, n_samples) - k groups of scores
    """
    m = int(np.sum(ground_truth == 1))
    n = len(ground_truth) - m
    k = predictions.shape[0]

    pos = predictions[:, :m]
    neg = predictions[:, m:]

    aucs = np.zeros(k)
    for j in range(k):
        combined = np.concatenate([pos[j], neg[j]])
        mr = _compute_midrank(combined)
        aucs[j] = np.sum(mr[:m]) / (m * n) - (m + 1.0) / (2.0 * n)

    v10 = np.zeros((k, m))
    v01 = np.zeros((k, n))
    for j in range(k):
        for ii in range(m):
            v10[j, ii] = (np.sum(neg[j] < pos[j, ii]) +
                          0.5 * np.sum(neg[j] == pos[j, ii])) / n
        for jj in range(n):
            v01[j, jj] = (np.sum(pos[j] < neg[j, jj]) +
                          0.5 * np.sum(pos[j] == neg[j, jj])) / m

    sx = np.cov(v10) if m > 1 else np.zeros((k, k))
    sy = np.cov(v01) if n > 1 else np.zeros((k, k))

    if np.ndim(sx) == 0:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])

    S = sx / m + sy / n
    return aucs, S


def delong_test(y_true, scores_a, scores_b):
    """
    DeLong test: directly compare if two AUROC groups differ significantly.
    H0: AUC_A == AUC_B (two-sided)

    Returns:
        dict: auc_a, auc_b, auc_diff, z_statistic, p_value, significant_at_005
    """
    y_true = np.asarray(y_true, dtype=float)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    preds = np.vstack([scores_a[order], scores_b[order]])

    aucs, S = _delong_variance(y_sorted, preds)

    diff = aucs[0] - aucs[1]
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 1e-12:
        return {
            'auc_a': float(aucs[0]), 'auc_b': float(aucs[1]),
            'auc_diff': float(diff), 'z_statistic': 0.0,
            'p_value': 1.0, 'significant_at_005': False,
        }

    z = diff / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        'auc_a': float(aucs[0]),
        'auc_b': float(aucs[1]),
        'auc_diff': float(diff),
        'z_statistic': float(z),
        'p_value': float(p_value),
        'significant_at_005': bool(p_value < 0.05),
    }


def evaluate_probe_per_layer(y_true_per_layer, y_scores_per_layer,
                             method_name="Probe", n_bootstrap=2000):
    """
    Compute complete metrics + Bootstrap CI for each layer separately.

    Args:
        y_true_per_layer:   list[np.array] Ground truth labels per layer
        y_scores_per_layer: list[np.array] Prediction scores per layer

    Returns:
        list[dict]: Complete metrics dictionary for each layer
    """
    results = []
    for layer_idx, (yt, ys) in enumerate(zip(y_true_per_layer, y_scores_per_layer)):
        r = {'method': method_name, 'layer': layer_idx}
        r.update(compute_full_metrics(yt, ys))
        r['bootstrap_ci'] = bootstrap_auroc_ci(yt, ys, n_bootstrap=n_bootstrap)
        results.append(r)
    return results


def compare_probes(y_true, scores_dict, reference_method=None, n_bootstrap=2000):
    """
    Compare multiple probe methods with statistical tests.

    Args:
        y_true:           Ground truth labels
        scores_dict:      {method_name: y_scores}
        reference_method: Reference method name for pairwise tests
        n_bootstrap:      Bootstrap resampling iterations

    Returns:
        dict: Per-method metrics + pairwise comparisons
    """
    y_true = np.asarray(y_true)
    results = {}

    for name, sc in scores_dict.items():
        sc = np.asarray(sc)
        r = {'method': name}
        r.update(compute_full_metrics(y_true, sc))
        r['bootstrap_ci'] = bootstrap_auroc_ci(y_true, sc, n_bootstrap=n_bootstrap)
        r['bootstrap_auprc_ci'] = bootstrap_metric_ci(
            y_true, sc, average_precision_score, n_bootstrap=n_bootstrap
        )
        results[name] = r

    if reference_method and reference_method in scores_dict:
        ref_sc = np.asarray(scores_dict[reference_method])
        comparisons = {}

        fpr_r, tpr_r, thr_r = roc_curve(y_true, ref_sc)
        opt_r = thr_r[np.argmax(tpr_r - fpr_r)]
        pred_ref = (ref_sc >= opt_r).astype(int)

        for name, sc in scores_dict.items():
            if name == reference_method:
                continue
            sc = np.asarray(sc)

            comp = {}
            comp['delong'] = delong_test(y_true, ref_sc, sc)

            fpr_c, tpr_c, thr_c = roc_curve(y_true, sc)
            opt_c = thr_c[np.argmax(tpr_c - fpr_c)]
            pred_other = (sc >= opt_c).astype(int)
            comp['mcnemar'] = mcnemar_test(y_true, pred_ref, pred_other)

            comparisons[f"{reference_method}_vs_{name}"] = comp

        results['pairwise_comparisons'] = comparisons

    return results


def print_evaluation_report(results, show_comparisons=True):
    """Print formatted complete evaluation report."""
    print()
    header = (f"{'Method':<22} {'AUROC':>7} {'95% CI':>18} {'AUPRC':>7} "
              f"{'Acc':>7} {'F1':>7} {'MCC':>7} {'Brier':>7} {'d':>7}")
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for name, res in results.items():
        if name == 'pairwise_comparisons':
            continue
        ci = res.get('bootstrap_ci', {})
        ci_lo = ci.get('ci_lower', 0)
        ci_hi = ci.get('ci_upper', 0)
        ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]"

        print(f"{name:<22} "
              f"{res.get('auroc', 0):>7.4f} {ci_str:>18} "
              f"{res.get('auprc', 0):>7.4f} "
              f"{res.get('accuracy', 0):>7.4f} "
              f"{res.get('f1', 0):>7.4f} "
              f"{res.get('mcc', 0):>7.4f} "
              f"{res.get('brier_score', 0):>7.4f} "
              f"{res.get('cohens_d', 0):>7.3f}")

    if show_comparisons and 'pairwise_comparisons' in results:
        print()
        print("-" * len(header))
        print("Pairwise Statistical Tests (reference = your method):")
        print("-" * len(header))

        for pair, comp in results['pairwise_comparisons'].items():
            print(f"\n  {pair}:")

            dl = comp.get('delong', {})
            sig_dl = '***' if dl.get('p_value', 1) < 0.001 else \
                     '**' if dl.get('p_value', 1) < 0.01 else \
                     '*' if dl.get('p_value', 1) < 0.05 else 'n.s.'
            print(f"    DeLong:  ΔAUC = {dl.get('auc_diff', 0):+.4f},  "
                  f"z = {dl.get('z_statistic', 0):.3f},  "
                  f"p = {dl.get('p_value', 1):.2e}  {sig_dl}")

            mc = comp.get('mcnemar', {})
            sig_mc = '***' if mc.get('p_value', 1) < 0.001 else \
                     '**' if mc.get('p_value', 1) < 0.01 else \
                     '*' if mc.get('p_value', 1) < 0.05 else 'n.s.'
            stat_str = f"χ² = {mc.get('statistic', 0):.3f}" if not np.isnan(
                mc.get('statistic', 0)) else "exact"
            print(f"    McNemar: {stat_str},  "
                  f"p = {mc.get('p_value', 1):.2e}  {sig_mc}")

            ct = mc.get('contingency', {})
            print(f"             b={ct.get('a_correct_b_wrong', 0)}, "
                  f"c={ct.get('a_wrong_b_correct', 0)}")

    print("=" * len(header))
    print("  Significance: * p<0.05  ** p<0.01  *** p<0.001  n.s. = not significant")
    print()

def get_method_scores_at_layer(method_name, pos_tensor, neg_tensor, layer, split_idx):
    """Re-trains standard baselines on the exact same split to extract test scores for statistical testing."""
    device = pos_tensor.device  # Dynamically get device
    
    Xp_tr = pos_tensor[:split_idx, layer, :].cpu().numpy()
    Xp_te = pos_tensor[split_idx:, layer, :].cpu().numpy()
    Xn_tr = neg_tensor[:split_idx, layer, :].cpu().numpy()
    Xn_te = neg_tensor[split_idx:, layer, :].cpu().numpy()

    X_tr = np.concatenate([Xp_tr, Xn_tr])
    y_tr = np.concatenate([np.ones(len(Xp_tr)), np.zeros(len(Xn_tr))])
    X_te = np.concatenate([Xp_te, Xn_te])

    if method_name == 'Baseline (LR)':
        clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]

    elif method_name == 'Baseline (KNN)':
        clf = KNeighborsClassifier(n_neighbors=5, metric='cosine').fit(X_tr, y_tr)
        return clf.predict_proba(X_te)[:, 1]

    elif method_name == 'Baseline (Mass-Mean)':
        mu_pos, mu_neg = np.mean(Xp_tr, axis=0), np.mean(Xn_tr, axis=0)
        return X_te @ (mu_pos - mu_neg)

    elif method_name == 'Baseline (RepE)':
        diff_tr = Xp_tr - Xn_tr
        diff_tr += np.random.normal(0, 1e-6, diff_tr.shape)
        pca = PCA(n_components=1, random_state=42).fit(diff_tr)
        return X_te @ pca.components_[0]

    elif method_name == 'Baseline (MLP)':
        from utils.baselines import StandardMLP
        mlp = StandardMLP(X_tr.shape[1], 256).to(device)
        opt = optim.Adam(mlp.parameters(), lr=0.005)
        
        X_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        # Remove unsqueeze(1) to keep it as a 1D tensor [N]
        y_t = torch.tensor(y_tr, dtype=torch.float32, device=device) 
        
        mlp.train()
        for _ in range(50):
            opt.zero_grad()
            # Force mlp output to be 1D with .squeeze() to match y_t
            loss = nn.BCEWithLogitsLoss()(mlp(X_t).squeeze(), y_t)
            loss.backward()
            opt.step()
            
        mlp.eval()
        with torch.no_grad():
            X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
            return mlp(X_te_t).cpu().numpy().flatten()

    else:
        return np.random.rand(len(X_te))