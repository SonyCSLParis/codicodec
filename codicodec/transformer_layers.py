# adapted from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
import torch
from torch import nn


def exists(val):
    return val is not None

def zero_init(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def init(module):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0.)
    return module


class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True, bias=False):
        super().__init__()
        self.eps = 1e-4
        if affine:
            self.g = nn.Parameter(torch.ones(dim))
            if bias:
                self.b = nn.Parameter(torch.zeros(dim))
            else:
                self.b = 0
        else:
            self.g = 1
            self.b = 0

    def forward(self, x):
        x = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return (x * self.g) + self.b
    

class LayerNorm(nn.LayerNorm):
    def __init__(self, dim, affine=True, bias=False):
        super().__init__(dim, eps=1e-4, elementwise_affine=False)
        if affine:
            self.g = nn.Parameter(torch.ones(dim))
            if bias:
                self.b = nn.Parameter(torch.zeros(dim))
            else:
                self.b = 0
        else:
            self.g = 1
            self.b = 0

    def forward(self, x):
        x = super().forward(x.float()).type_as(x)
        x = (x * self.g) + self.b
        return x


class Feedforward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mlp_mult)
        dim_out = dim

        self.activation = nn.SiLU()
        self.to_mlp = init(nn.Linear(dim, inner_dim, bias=False))
        self.to_out = zero_init(nn.Linear(inner_dim//2, dim_out, bias=False))
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = self.to_mlp(x)
        x1,x2 = x.chunk(2, dim=-1)
        x = self.activation(x1) * x2
        x = self.do(x)
        x = self.to_out(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads = 4, group_together = 1, training_length = None, causal=False, pos_emb='none'):
        super().__init__()
        self.dim_head = embed_dim//num_heads
        self.heads = num_heads
        hidden_dim = self.dim_head * num_heads
        self.hidden_dim = hidden_dim
        self.training_length = training_length
        self.causal = causal

        if group_together is None:
            group_together = 1
        self.group_together = group_together

        self.to_qkv = init(nn.Linear(embed_dim, hidden_dim*3, bias=False))
        self.to_out = zero_init(nn.Linear(hidden_dim, embed_dim, bias=False))
        self.q_norm = RMSNorm(self.dim_head, affine=True)
        self.k_norm = RMSNorm(self.dim_head, affine=True)

    def forward(self, x, num_groups=None, attn_mask=None):
        b, n, _ = x.size()
        x = self.to_qkv(x)
        q, k, v = torch.chunk(x, chunks=3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1,2), (q,k,v))
        q = self.q_norm(q)
        k = self.k_norm(k)
        scale_factor = 1./(float(self.dim_head) ** 0.5)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal if not exists(attn_mask) else False, scale=scale_factor, attn_mask=attn_mask)
        out = out.transpose(1,2).contiguous().view(b, n, self.hidden_dim)
        return self.to_out(out)


class AdaptiveNorm(nn.Module):
    def __init__(self, dim, cond_dim=None):
        super(AdaptiveNorm, self).__init__()
        self.norm = LayerNorm(dim, affine=not exists(cond_dim), bias=False)
        if exists(cond_dim):
            self.cond_proj = zero_init(nn.Linear(cond_dim, dim))

    def forward(self, x, cond=None):
        x = self.norm(x)
        if exists(cond):
            cond = self.cond_proj(cond)
            if len(cond.shape) == 2:
                cond = cond.unsqueeze(-2)
            if x.shape[0]>cond.shape[0]:
                cond = cond.repeat(x.shape[0]//cond.shape[0],1,1)
            if cond.shape[-2]==2:
                cond1, cond2 = torch.chunk(cond, 2, dim=-2)
                x1, x2 = torch.chunk(x, 2, dim=-2)
                x = torch.cat((x1 * (1.+cond1), x2 * (1.+cond2)), dim=-2)
            else:
                x = x * (1.+cond)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, cond_dim=None, group_together=1, training_length=None, causal=False, pos_emb='none'):
        super(Attention, self).__init__()
        self.mha = MultiHeadAttention(embed_dim=dim, num_heads=heads, group_together=group_together, training_length=training_length, causal=causal, pos_emb=pos_emb)
        self.norm = AdaptiveNorm(dim, cond_dim)

    def forward(self, x, cond=None, num_groups=None, attn_mask=None):
        inp = x
        x = self.norm(x, cond)
        x = self.mha(x, num_groups=num_groups, attn_mask=attn_mask)
        return x+inp


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult=4, cond_dim=None, dropout=0.):
        super(MLP, self).__init__()
        self.ff = Feedforward(dim=dim, mlp_mult=mlp_mult, dropout=dropout)
        self.norm = AdaptiveNorm(dim, cond_dim)

    def forward(self, x, cond=None, attn_mask=None):
        inp = x
        x = self.norm(x, cond)
        x = self.ff(x)
        return x+inp


class AttentionBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_mult=4, cond_dim=None, group_together=1, training_length=None, causal=False, pos_emb='none', dropout=0.):
        super(AttentionBlock, self).__init__()
        self.attn = Attention(dim, heads, cond_dim=cond_dim, group_together=group_together, training_length=training_length, causal=causal, pos_emb=pos_emb)
        self.mlp = MLP(dim, mlp_mult, cond_dim=cond_dim, dropout=dropout)
        
    def forward(self, x, cond=None, num_groups=None, attn_mask=None):
        x = self.attn(x, cond, num_groups=num_groups, attn_mask=attn_mask)
        x = self.mlp(x, cond)
        return x