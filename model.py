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
        #keys = self.W_key.T.matmul(x.T).T
        print("keys shape")
        print(keys.shape)
        queries = x.matmul(self.W_query)
        #queries = self.W_query.T.matmul(x.T).T
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
        #export_to_csv(attn_weights.detach())
        print("attn weights shape")
        print(attn_weights.shape)
        #print(torch.sum(attn_weights,dim=1))

        context_vex = attn_weights.matmul(values)
        #print("context vex sum")
        #print(torch.sum(context_vex, dim=1))
        return context_vex

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)  # dim=-1, the last dimension
