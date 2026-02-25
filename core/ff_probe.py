# core/ff_probe.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FFLayerProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, lr: float = 0.005, device: str = 'cuda'):
        super().__init__()
        self.device = device
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False).to(device)
        nn.init.orthogonal_(self.linear.weight)
        
        self.dropout = nn.Dropout(p=0.2)
        
        # Threshold changed to 0.0 because energy will be Z-score normalized (mean=0)
        self.threshold = nn.Parameter(torch.tensor([0.0], device=device))
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instance-wise centering
        x_centered = x - x.mean(dim=1, keepdim=True) 
        x_norm = F.normalize(x_centered, p=2, dim=1) 
        
        h = self.linear(x_norm)
        
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
        
        g_pos_raw = self(x_pos)
        g_neg_raw = self(x_neg)
        
        # --- Adaptive Energy Normalization (CRITICAL FIX) ---
        # Forces the absolute energy scales of SQA and LQA to match
        # by standardizing them to Mean=0, Std=1 within the batch.
        g_all = torch.cat([g_pos_raw, g_neg_raw])
        g_mean = g_all.mean()
        g_std = g_all.std() + 1e-6
        
        g_pos = (g_pos_raw - g_mean) / g_std
        g_neg = (g_neg_raw - g_mean) / g_std
        
        # Standard losses (Temperature removed)
        logits_pos = g_pos - self.threshold
        logits_neg = g_neg - self.threshold
        
        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, torch.ones_like(logits_pos))
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, torch.zeros_like(logits_neg))
        
        # Delta=1.0 is now meaningful and stable for BOTH domains
        delta = 1.0 
        loss_margin = F.relu(delta - (g_pos - g_neg)).mean()
        
        loss = 1.0 * loss_margin + 0.5 * (loss_pos + loss_neg)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()