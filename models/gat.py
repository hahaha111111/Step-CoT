import torch
import torch.nn as nn
import torch.nn.functional as F

class GATConv(nn.Module):
    # Same as in original script, copied here
    def __init__(self, in_dim, out_dim, heads=4, concat=True, dropout=0.1, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(heads, out_dim))
        self.a_dst = nn.Parameter(torch.Tensor(heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.bias)

    def forward(self, h, mask=None):
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.out_dim).permute(0,2,1,3)
        a_src = self.a_src.view(1, self.heads, 1, self.out_dim)
        a_dst = self.a_dst.view(1, self.heads, 1, self.out_dim)
        el = (Wh * a_src).sum(dim=-1)  # (B, heads, N)
        er = (Wh * a_dst).sum(dim=-1)
        e = self.leaky_relu(el.unsqueeze(-1) + er.unsqueeze(-2))

        if mask is not None:
            mask_j = mask.unsqueeze(1).unsqueeze(2)
            e = e.masked_fill(~mask_j, -1e9)

        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        h_prime = torch.matmul(alpha, Wh)
        if self.concat:
            h_prime = h_prime.permute(0,2,1,3).contiguous().view(B, N, self.heads * self.out_dim)
        else:
            h_prime = h_prime.mean(dim=1)
        return h_prime + self.bias

class StackedGAT(nn.Module):
    def __init__(self, in_dim, hid_out_per_head=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        cur_in = in_dim
        for i in range(layers):
            self.layers.append(GATConv(cur_in, hid_out_per_head, heads=heads, concat=True, dropout=dropout))
            out_dim = hid_out_per_head * heads
            self.projs.append(nn.Linear(cur_in, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            cur_in = out_dim
        self.out_dim = cur_in

    def forward(self, memory, mask=None):
        x = memory
        for i, gat in enumerate(self.layers):
            out = gat(x, mask=mask)
            res = self.projs[i](x)
            x = self.norms[i](out + res)
            x = F.elu(x)
        return x