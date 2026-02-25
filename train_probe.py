# train_probe.py
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from config import DATA_LIMIT, HIDDEN_DIM, get_feature_path, get_checkpoint_path
from core import FFLayerProbe

# === Configuration ===
FEATURES_PATH = get_feature_path("simpleqa")
CHECKPOINTS_DIR = './checkpoints'

def calculate_ff_auroc(pos_tensor, neg_tensor, n_epochs=50, lr=0.005, device='cuda'):
    n_layers = pos_tensor.shape[1]
    input_dim = pos_tensor.shape[2]
    batch_size = 64
    n_repeats = 3
    test_ratio = 0.3

    aurocs = []
    trained_models = {}
    
    pos_tensor = pos_tensor.to(device)
    neg_tensor = neg_tensor.to(device)
    N = pos_tensor.shape[0]

    for i in tqdm(range(n_layers), desc="FF Probe Training"):
        X_pos = pos_tensor[:, i, :]
        X_neg = neg_tensor[:, i, :]

        repeat_aucs = []
        best_auc, best_state = -1.0, None

        for _ in range(n_repeats):
            perm = torch.randperm(N, device=device)
            split = int((1 - test_ratio) * N)
            tr_idx, te_idx = perm[:split], perm[split:]

            Xp_tr, Xp_te = X_pos[tr_idx], X_pos[te_idx]
            Xn_tr, Xn_te = X_neg[tr_idx], X_neg[te_idx]

            probe = FFLayerProbe(input_dim, hidden_dim=HIDDEN_DIM, lr=lr, device=device)
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

            repeat_aucs.append(auc)

            if auc > best_auc:
                best_auc = auc
                best_state = {
                    'threshold': probe.threshold.item(),
                    'state_dict': {k: v.cpu().clone() for k, v in probe.state_dict().items()}
                }

        mean_auc = float(np.mean(repeat_aucs))
        aurocs.append(mean_auc)
        trained_models[i] = best_state

        if i % 5 == 0:
            print(f"  Layer {i}: AUC={mean_auc:.4f}±{np.std(repeat_aucs):.4f} "
                  f"(θ={best_state['threshold']:.2f})")

    return aurocs, trained_models

if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Features not found at {FEATURES_PATH}. Run extract_features.py first.")
        exit(1)
        
    print(f"📥 Loading features from {FEATURES_PATH}...")
    data = torch.load(FEATURES_PATH)
    qa_pos_tensor = data["pos"]
    qa_neg_tensor = data["neg"]
    print(f"✅ Loaded features shape: {qa_pos_tensor.shape}")

    results = {}
    ff_directions = {}

    print("\n🚀 Training Forward-Forward Probes...")
    ff_runs = []
    
    for seed in range(3):
        torch.manual_seed(42 + seed)
        print(f"\n--- Run {seed + 1}/3 ---")
        aucs, dirs = calculate_ff_auroc(qa_pos_tensor, qa_neg_tensor)
        ff_runs.append(aucs)
        
        if seed == 0:
            ff_directions = dirs 

    results['FF_SimpleQA_Mean'] = np.mean(ff_runs, axis=0)
    results['FF_SimpleQA_Std']  = np.std(ff_runs, axis=0)

    print("\n✅ Training Complete.")
    
    res_path = get_checkpoint_path("metrics")
    weights_path = get_checkpoint_path("weights")
    
    torch.save(results, res_path)
    torch.save(ff_directions, weights_path)
    print(f"💾 Metrics saved to: {res_path}")
    print(f"💾 Weights saved to: {weights_path}")