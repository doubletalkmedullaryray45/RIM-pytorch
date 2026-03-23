from __future__ import annotations
from functools import partial

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear

from einops import einsum
from einops.layers.torch import Rearrange

# constants

LinearNoBias = partial(Linear, bias = False)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.to_qkv = LinearNoBias(dim, dim_inner * 3)

        self.to_out = LinearNoBias(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        tokens
    ):
        q, k, v = self.to_qkv(tokens).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

# block

class TransformerBlock(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()

    def forward(
        self,
        tokens
    ):
        return tokens

# classes

class DepthlessTransformer(Module):
    def __init__(
        self
    ):
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        tokens
    ):
        return tokens
