# utils/baselines.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def calculate_lr_auroc(pos_tensor, neg_tensor, test_size=0.5, random_state=42):
    n_layers = pos_tensor.shape[1]
    aurocs = []
    
    print("\nTraining LR Baseline...", end="\r")
    for i in range(n_layers):
        X_pos = pos_tensor[:, i, :].cpu().numpy()
        X_neg = neg_tensor[:, i, :].cpu().numpy()
        
        X = np.concatenate([X_pos, X_neg])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        clf = LogisticRegression(solver='liblinear', max_iter=500).fit(X_train, y_train)
        probs = clf.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
        aurocs.append(auc)
        
    return aurocs

def calculate_mass_mean_auroc(pos_tensor, neg_tensor, test_size=0.5, random_state=42):
    n_layers = pos_tensor.shape[1]
    aurocs = []
    
    print("\nCalculating Mass-Mean Baseline...", end="\r")
    for i in range(n_layers):
        X_pos = pos_tensor[:, i, :].cpu().numpy()
        X_neg = neg_tensor[:, i, :].cpu().numpy()
        
        Xp_tr, Xp_te = train_test_split(X_pos, test_size=test_size, random_state=random_state)
        Xn_tr, Xn_te = train_test_split(X_neg, test_size=test_size, random_state=random_state)
        
        mu_pos = np.mean(Xp_tr, axis=0)
        mu_neg = np.mean(Xn_tr, axis=0)
        direction = mu_pos - mu_neg
        
        X_te = np.concatenate([Xp_te, Xn_te])
        y_te = np.concatenate([np.ones(len(Xp_te)), np.zeros(len(Xn_te))])
        
        scores = np.dot(X_te, direction)
        
        try:
            auc = roc_auc_score(y_te, scores)
        except ValueError:
            auc = 0.5
        aurocs.append(auc)
        
    return aurocs

class StandardMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1) 
        )
        
    def forward(self, x):
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        return self.net(x_norm).squeeze(-1)

def calculate_mlp_auroc(pos_tensor, neg_tensor, n_epochs=50, lr=0.005, device='cuda'):
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    batch_size = 64
    n_repeats = 3
    test_ratio = 0.3

    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="MLP Baseline Training"):
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
            y_tr = torch.cat([torch.ones(len(Xp_tr), device=device), torch.zeros(len(Xn_tr), device=device)])
            
            mlp = StandardMLP(input_dim, hidden_dim=256).to(device)
            optimizer = optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-3)
            criterion = nn.BCEWithLogitsLoss()

            for _ in range(n_epochs):
                mlp.train()
                perm_batch = torch.randperm(len(X_tr), device=device)
                for s in range(0, len(X_tr), batch_size):
                    idx = perm_batch[s:s+batch_size]
                    optimizer.zero_grad()
                    logits = mlp(X_tr[idx])
                    loss = criterion(logits, y_tr[idx])
                    loss.backward()
                    optimizer.step()

            mlp.eval()
            with torch.no_grad():
                logits_p = mlp(Xp_te).cpu().numpy()
                logits_n = mlp(Xn_te).cpu().numpy()

            y_true = np.concatenate([np.ones(len(logits_p)), np.zeros(len(logits_n))])
            y_scores = np.concatenate([logits_p, logits_n])
            try:
                auc = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc = 0.5
            repeat_aucs.append(auc)

        aurocs.append(float(np.mean(repeat_aucs)))
        
    return aurocs

def calculate_ccs_auroc(pos_tensor, neg_tensor, n_epochs=200, lr=0.005, device='cuda', test_ratio=0.3):
    """Contrast-Consistent Search (CCS) Baseline."""
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="CCS Baseline Training"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]

        perm = torch.randperm(N, device=device)
        split = int((1 - test_ratio) * N)
        tr_idx, te_idx = perm[:split], perm[split:]

        Xp_tr, Xp_te = X_pos[tr_idx], X_pos[te_idx]
        Xn_tr, Xn_te = X_neg[tr_idx], X_neg[te_idx]

        mu = torch.cat([Xp_tr, Xn_tr]).mean(dim=0, keepdim=True)
        std = torch.cat([Xp_tr, Xn_tr]).std(dim=0, keepdim=True) + 1e-6
        Xp_tr = (Xp_tr - mu) / std
        Xn_tr = (Xn_tr - mu) / std
        Xp_te = (Xp_te - mu) / std
        Xn_te = (Xn_te - mu) / std

        probe = nn.Linear(input_dim, 1).to(device)
        
        with torch.no_grad():
            diff = (Xp_tr.mean(dim=0) - Xn_tr.mean(dim=0))
            probe.weight.copy_(diff.unsqueeze(0))
            probe.bias.fill_(0.0)

        optimizer = optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)

        probe.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            p_pos = torch.sigmoid(probe(Xp_tr).squeeze(-1))
            p_neg = torch.sigmoid(probe(Xn_tr).squeeze(-1))
            
            consistency_loss = ((p_pos + p_neg - 1.0) ** 2).mean()
            informative_loss = (torch.min(p_pos, p_neg) ** 2).mean()
            loss = consistency_loss + informative_loss
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            logits_p = probe(Xp_te).squeeze(-1).cpu().numpy()
            logits_n = probe(Xn_te).squeeze(-1).cpu().numpy()

        y_true = np.concatenate([np.ones(len(logits_p)), np.zeros(len(logits_n))])
        y_scores = np.concatenate([logits_p, logits_n])
        try:
            auc = roc_auc_score(y_true, y_scores)
            if auc < 0.5:
                auc = 1.0 - auc
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs

def calculate_prob_entropy_auroc(pos_scores, neg_scores):
    """
    Probability / Semantic Entropy Baseline.
    Requires 1D arrays of model probabilities/entropies, not hidden states.
    Returns a single AUROC float.
    """
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        if auc < 0.5:
            auc = 1.0 - auc
    except ValueError:
        auc = 0.5
        
    return float(auc)

class FFLayerProbeNoPeerNorm(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, lr=0.01, device='cuda'):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(in_dim, hidden_dim).to(device)
        nn.init.orthogonal_(self.linear.weight)
        self.dropout = nn.Dropout(p=0.1)
        self.threshold = nn.Parameter(torch.tensor([2.0], device=device))
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-3)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1) 
        h = self.linear(x_norm)
        h = F.relu(h) 
        h = self.dropout(h)
        scale_factor = h.shape[1] ** 0.5
        goodness = h.pow(2).sum(dim=1) / scale_factor
        return goodness

    def train_step(self, x_pos, x_neg):
        self.train()
        self.optimizer.zero_grad()
        g_pos = self(x_pos)
        g_neg = self(x_neg)
        
        logits_pos = g_pos - self.threshold
        logits_neg = g_neg - self.threshold
        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
        
        loss_margin = F.relu(1.0 - (g_pos - g_neg)).mean()
        loss = loss_pos + 2.0 * loss_neg + 0.5 * loss_margin
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

def calculate_ff_no_peernorm_auroc(pos_tensor, neg_tensor, n_epochs=50, lr=0.01, device='cuda'):
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    batch_size = 64

    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="Ablation (No PeerNorm) Training"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]
        
        perm = torch.randperm(N, device=device)
        split = int(0.7 * N)
        tr_idx, te_idx = perm[:split], perm[split:]

        Xp_tr, Xp_te = X_pos[tr_idx], X_pos[te_idx]
        Xn_tr, Xn_te = X_neg[tr_idx], X_neg[te_idx]

        probe = FFLayerProbeNoPeerNorm(input_dim, hidden_dim=256, lr=lr, device=device)
        n_tr = min(Xp_tr.size(0), Xn_tr.size(0))

        for _ in range(n_epochs):
            pp = torch.randperm(Xp_tr.size(0), device=device)
            np_ = torch.randperm(Xn_tr.size(0), device=device)
            for s in range(0, n_tr, batch_size):
                e = min(s + batch_size, n_tr)
                probe.train_step(Xp_tr[pp[s:e]], Xn_tr[np_[s:e]])

        with torch.no_grad():
            probe.eval()
            g_pos = probe(Xp_te).cpu().numpy()
            g_neg = probe(Xn_te).cpu().numpy()

        y_true = np.concatenate([np.ones(len(g_pos)), np.zeros(len(g_neg))])
        y_scores = np.concatenate([g_pos, g_neg])
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.5
        aurocs.append(auc)

    return aurocs

class FFLayerProbeNoZScore(nn.Module):
    """Ablation model: FF Probe without Adaptive Z-Score Normalization."""
    def __init__(self, in_dim, hidden_dim=256, lr=0.01, device='cuda'):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        nn.init.orthogonal_(self.linear.weight)
        
        self.dropout = nn.Dropout(p=0.2)
        
        # Threshold is set to 2.0 (or learnable) instead of 0.0 
        # because raw goodness is always positive and un-normalized here
        self.threshold = nn.Parameter(torch.tensor([2.0], device=device))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance-wise centering is KEPT
        x_centered = x - x.mean(dim=1, keepdim=True) 
        x_norm = F.normalize(x_centered, p=2, dim=1) 
        
        h = self.linear(x_norm)
        
        # PeerNorm is KEPT
        h_mean = h.mean(dim=1, keepdim=True)
        h = F.relu(h - h_mean)
        h = self.dropout(h)
        
        # Raw Energy calculation
        scale_factor = h.shape[1] ** 0.5
        raw_goodness = h.pow(2).sum(dim=1) / scale_factor
        
        return raw_goodness
    
    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> float:
        self.train()
        self.optimizer.zero_grad()
        
        g_pos = self(x_pos)
        g_neg = self(x_neg)
        
        # ABLATION: Adaptive Energy Normalization (Z-score) is REMOVED
        # We directly use raw goodness scores for logits and margin
        
        logits_pos = g_pos - self.threshold
        logits_neg = g_neg - self.threshold
        
        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
        
        delta = 1.0 
        loss_margin = F.relu(delta - (g_pos - g_neg)).mean()
        
        loss = 1.0 * loss_margin + 0.5 * (loss_pos + loss_neg)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

def calculate_ff_no_zscore_auroc(pos_tensor, neg_tensor, n_epochs=50, lr=0.005, device='cuda'):
    """Trains the ablated FF probe (No Z-Score) and returns AUROC per layer."""
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    batch_size = 64

    aurocs = []
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="Ablation (No Z-Score) Training"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]
        
        perm = torch.randperm(N, device=device)
        split = int(0.7 * N)
        tr_idx, te_idx = perm[:split], perm[split:]

        Xp_tr, Xp_te = X_pos[tr_idx], X_pos[te_idx]
        Xn_tr, Xn_te = X_neg[tr_idx], X_neg[te_idx]

        probe = FFLayerProbeNoZScore(input_dim, hidden_dim=256, lr=lr, device=device)
        n_tr = min(Xp_tr.size(0), Xn_tr.size(0))

        for _ in range(n_epochs):
            pp = torch.randperm(Xp_tr.size(0), device=device)
            np_ = torch.randperm(Xn_tr.size(0), device=device)
            for s in range(0, n_tr, batch_size):
                e = min(s + batch_size, n_tr)
                probe.train_step(Xp_tr[pp[s:e]], Xn_tr[np_[s:e]])

        with torch.no_grad():
            probe.eval()
            g_pos = probe(Xp_te).cpu().numpy()
            g_neg = probe(Xn_te).cpu().numpy()

        y_true = np.concatenate([np.ones(len(g_pos)), np.zeros(len(g_neg))])
        y_scores = np.concatenate([g_pos, g_neg])
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.5
        aurocs.append(float(auc))

    return aurocs