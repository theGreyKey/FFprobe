# ============================================================
# utils/advanced_baselines.py
# Frontier Probes Baselines
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tqdm import tqdm


# ================================================================
# 1. KNN Probe (Non-parametric Baseline)
# ================================================================
def calculate_knn_auroc(pos_tensor, neg_tensor, k=5, test_size=0.3, random_state=42):
    """K-Nearest Neighbors probe baseline using cosine distance."""
    n_layers = pos_tensor.shape[1]
    aurocs = []

    for i in tqdm(range(n_layers), desc="KNN Probe"):
        X_pos = pos_tensor[:, i, :].cpu().numpy()
        X_neg = neg_tensor[:, i, :].cpu().numpy()

        X = np.concatenate([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        clf = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs


# ================================================================
# 2. Mahalanobis Distance Probe
# ================================================================
def calculate_mahalanobis_auroc(pos_tensor, neg_tensor, test_size=0.3, random_state=42):
    """Mahalanobis distance probe with shared covariance matrix and regularization."""
    n_layers = pos_tensor.shape[1]
    aurocs = []

    for i in tqdm(range(n_layers), desc="Mahalanobis Probe"):
        X_pos = pos_tensor[:, i, :].cpu().numpy()
        X_neg = neg_tensor[:, i, :].cpu().numpy()

        Xp_tr, Xp_te = train_test_split(X_pos, test_size=test_size, random_state=random_state)
        Xn_tr, Xn_te = train_test_split(X_neg, test_size=test_size, random_state=random_state)

        mu_pos = np.mean(Xp_tr, axis=0)
        mu_neg = np.mean(Xn_tr, axis=0)

        X_all_tr = np.concatenate([Xp_tr, Xn_tr])
        cov = np.cov(X_all_tr, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-4

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        X_te = np.concatenate([Xp_te, Xn_te])
        y_te = np.concatenate([np.ones(len(Xp_te)), np.zeros(len(Xn_te))])

        diff_pos = X_te - mu_pos
        diff_neg = X_te - mu_neg
        d_pos = np.sum(diff_pos @ cov_inv * diff_pos, axis=1)
        d_neg = np.sum(diff_neg @ cov_inv * diff_neg, axis=1)
        scores = d_neg - d_pos

        try:
            auc = roc_auc_score(y_te, scores)
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs


# ================================================================
# 3. LDA Probe (Linear Discriminant Analysis)
# ================================================================
def calculate_lda_auroc(pos_tensor, neg_tensor, test_size=0.3, random_state=42):
    """Classical LDA baseline with shrinkage regularization."""
    n_layers = pos_tensor.shape[1]
    aurocs = []

    for i in tqdm(range(n_layers), desc="LDA Probe"):
        X_pos = pos_tensor[:, i, :].cpu().numpy()
        X_neg = neg_tensor[:, i, :].cpu().numpy()

        X = np.concatenate([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs


# ================================================================
# 4. RepE Probe (Representation Engineering)
# ================================================================
def calculate_repe_auroc(pos_tensor, neg_tensor, n_components=1, test_size=0.3, random_state=42):
    """PCA-based directional probe on difference vectors (pos - neg)."""
    n_layers = pos_tensor.shape[1]
    aurocs = []

    for i in tqdm(range(n_layers), desc="RepE Probe"):
        X_pos = pos_tensor[:, i, :].cpu().numpy()
        X_neg = neg_tensor[:, i, :].cpu().numpy()

        N = min(len(X_pos), len(X_neg))
        X_pos_aligned, X_neg_aligned = X_pos[:N], X_neg[:N]

        indices = np.arange(N)
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
        split = int(N * (1 - test_size))
        tr_idx, te_idx = indices[:split], indices[split:]

        Xp_tr, Xp_te = X_pos_aligned[tr_idx], X_pos_aligned[te_idx]
        Xn_tr, Xn_te = X_neg_aligned[tr_idx], X_neg_aligned[te_idx]

        diff_tr = Xp_tr - Xn_tr
        diff_tr += np.random.normal(0, 1e-6, diff_tr.shape) # Add tiny noise to prevent zero-variance
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(diff_tr)
        direction = pca.components_[0]

        X_te = np.concatenate([Xp_te, Xn_te])
        y_te = np.concatenate([np.ones(len(Xp_te)), np.zeros(len(Xn_te))])
        scores = X_te @ direction

        try:
            auc = roc_auc_score(y_te, scores)
            if auc < 0.5:
                auc = 1.0 - auc
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs


# ================================================================
# 5. Nonlinear CCS (NL-CCS)
# ================================================================
class _NLCCSNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x).squeeze(-1))


def calculate_nlccs_auroc(pos_tensor, neg_tensor, n_epochs=50, lr=0.001,
                          device='cuda', test_ratio=0.3):
    """Nonlinear CCS with MLP replacing linear probe, consistency and informative losses."""
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="NL-CCS Training"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]

        perm = torch.randperm(N, device=device)
        split = int((1 - test_ratio) * N)
        tr_idx, te_idx = perm[:split], perm[split:]

        Xp_tr, Xp_te = X_pos[tr_idx], X_pos[te_idx]
        Xn_tr, Xn_te = X_neg[tr_idx], X_neg[te_idx]

        mu = torch.cat([Xp_tr, Xn_tr]).mean(0, keepdim=True)
        std = torch.cat([Xp_tr, Xn_tr]).std(0, keepdim=True) + 1e-6
        Xp_tr, Xn_tr = (Xp_tr - mu) / std, (Xn_tr - mu) / std
        Xp_te, Xn_te = (Xp_te - mu) / std, (Xn_te - mu) / std

        probe = _NLCCSNet(input_dim, hidden_dim=128).to(device)
        optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)

        probe.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            p_pos = probe(Xp_tr)
            p_neg = probe(Xn_tr)

            consistency = ((p_pos + p_neg - 1.0) ** 2).mean()
            informative = (torch.min(p_pos, p_neg) ** 2).mean()
            confidence = (torch.min(
                F.binary_cross_entropy(p_pos, torch.ones_like(p_pos), reduction='none'),
                F.binary_cross_entropy(p_pos, torch.zeros_like(p_pos), reduction='none'),
            )).mean()

            loss = consistency + 0.5 * informative + 0.5 * confidence
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            s_pos = probe(Xp_te).cpu().numpy()
            s_neg = probe(Xn_te).cpu().numpy()

        y_true = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
        y_scores = np.concatenate([s_pos, s_neg])
        try:
            auc = roc_auc_score(y_true, y_scores)
            if auc < 0.5:
                auc = 1.0 - auc
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs


# ================================================================
# 6. SAPLMA-style Multi-layer Supervised Probe
# ================================================================
class _SAPLMANet(nn.Module):
    def __init__(self, in_dim, hidden_dims=(256, 128)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def calculate_saplma_auroc(pos_tensor, neg_tensor, n_epochs=50, lr=0.001,
                           device='cuda', test_ratio=0.3, n_repeats=3):
    """SAPLMA-style multi-layer supervised probe with cosine annealing."""
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    batch_size = 64
    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="SAPLMA Probe Training"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]
        repeat_aucs = []

        for _ in range(n_repeats):
            perm = torch.randperm(N, device=device)
            split = int((1 - test_ratio) * N)
            tr_idx, te_idx = perm[:split], perm[split:]

            Xp_tr, Xp_te = X_pos[tr_idx], X_pos[te_idx]
            Xn_tr, Xn_te = X_neg[tr_idx], X_neg[te_idx]

            X_tr = torch.cat([Xp_tr, Xn_tr])
            y_tr = torch.cat([torch.ones(len(Xp_tr), device=device),
                              torch.zeros(len(Xn_tr), device=device)])

            probe = _SAPLMANet(input_dim).to(device)
            optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(n_epochs):
                probe.train()
                perm_b = torch.randperm(len(X_tr), device=device)
                for s in range(0, len(X_tr), batch_size):
                    idx = perm_b[s:s + batch_size]
                    optimizer.zero_grad()
                    loss = criterion(probe(X_tr[idx]), y_tr[idx])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()

            probe.eval()
            with torch.no_grad():
                lp = probe(Xp_te).cpu().numpy()
                ln = probe(Xn_te).cpu().numpy()

            y_true = np.concatenate([np.ones(len(lp)), np.zeros(len(ln))])
            y_sc = np.concatenate([lp, ln])
            try:
                auc = roc_auc_score(y_true, y_sc)
            except ValueError:
                auc = 0.5
            repeat_aucs.append(auc)

        aurocs.append(float(np.mean(repeat_aucs)))

    return aurocs


# ================================================================
# 7. Concept Bottleneck Probe
# ================================================================
class _ConceptBottleneckNet(nn.Module):
    def __init__(self, in_dim, n_concepts=32, sparsity=0.1):
        super().__init__()
        self.concept_layer = nn.Linear(in_dim, n_concepts, bias=False)
        nn.init.orthogonal_(self.concept_layer.weight)
        self.classifier = nn.Linear(n_concepts, 1)
        self.sparsity = sparsity

    def forward(self, x):
        x = F.normalize(x, p=2, dim=1)
        concepts = torch.sigmoid(self.concept_layer(x))
        return self.classifier(concepts).squeeze(-1), concepts

    def sparsity_loss(self, concepts):
        return (concepts.mean(dim=0) - self.sparsity).pow(2).mean()


def calculate_concept_bottleneck_auroc(pos_tensor, neg_tensor, n_concepts=32,
                                       n_epochs=50, lr=0.003, device='cuda',
                                       test_ratio=0.3, n_repeats=3):
    """Interpretable bottleneck layer with orthogonal projection and sparsity constraint."""
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    batch_size = 64
    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="Concept Bottleneck Probe"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]
        repeat_aucs = []

        for _ in range(n_repeats):
            perm = torch.randperm(N, device=device)
            split = int((1 - test_ratio) * N)
            tr_idx, te_idx = perm[:split], perm[split:]

            X_tr = torch.cat([X_pos[tr_idx], X_neg[tr_idx]])
            y_tr = torch.cat([torch.ones(len(tr_idx), device=device),
                              torch.zeros(len(tr_idx), device=device)])

            probe = _ConceptBottleneckNet(input_dim, n_concepts).to(device)
            optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(n_epochs):
                probe.train()
                pp = torch.randperm(len(X_tr), device=device)
                for s in range(0, len(X_tr), batch_size):
                    idx = pp[s:s + batch_size]
                    optimizer.zero_grad()
                    logits, concepts = probe(X_tr[idx])
                    loss = criterion(logits, y_tr[idx]) + 0.1 * probe.sparsity_loss(concepts)
                    loss.backward()
                    optimizer.step()

            probe.eval()
            with torch.no_grad():
                lp, _ = probe(X_pos[te_idx])
                ln, _ = probe(X_neg[te_idx])
                lp, ln = lp.cpu().numpy(), ln.cpu().numpy()

            y_true = np.concatenate([np.ones(len(lp)), np.zeros(len(ln))])
            y_sc = np.concatenate([lp, ln])
            try:
                auc = roc_auc_score(y_true, y_sc)
            except ValueError:
                auc = 0.5
            repeat_aucs.append(auc)

        aurocs.append(float(np.mean(repeat_aucs)))

    return aurocs
