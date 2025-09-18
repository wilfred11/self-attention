import torch
from torch import nn

torch.manual_seed(123)

import numpy as np

def export_to_csv(t):
    np.savetxt('exports/att.txt',t.numpy())


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v):
        super().__init__()
        self.d_in=d_in
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        keys = x.matmul(self.W_key)
        print("keys shape")
        print(keys.shape)
        queries = x.matmul(self.W_query)
        print("queries shape")
        print(queries.shape)
        values = x.matmul(self.W_value)

        # unnormalized attention weights
        attn_scores = queries.matmul(keys.T)
        print("attention scores shape")
        print(attn_scores.shape)
        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq ** 0.5, dim=-1
        )
        export_to_csv(attn_weights.detach())
        print(attn_weights)
        print(torch.sum(attn_weights,dim=1))

        context_vex = attn_weights.matmul(values)
        return context_vex
