import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_step_criteria(counts, device):
    """Create weighted CrossEntropyLoss per step."""
    criteria = []
    for i, c in enumerate(counts):
        freq = c.astype(np.float32) + 1e-6
        inv = 1.0 / freq
        weights = inv / inv.sum() * len(inv)
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        criteria.append(nn.CrossEntropyLoss(weight=w))
    return criteria

def H_ch(u, v):
    """Centered HSIC loss (simplified)."""
    m_u = torch.mm(u, u.t())
    m_v = torch.mm(v, v.t())
    n_u = m_u.shape[0]
    n_v = m_v.shape[0]
    c_u = torch.eye(n_u, device=u.device) - torch.ones((n_u, n_u), device=u.device) / float(n_u)
    c_v = torch.eye(n_v, device=v.device) - torch.ones((n_v, n_v), device=v.device) / float(n_v)
    tr_u = torch.mm(torch.mm(c_u, m_u), c_u)
    tr_v = torch.mm(torch.mm(c_v, m_v), c_v)
    return torch.sum(tr_u * tr_v)