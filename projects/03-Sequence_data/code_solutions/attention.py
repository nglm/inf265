import torch
import math
from torch import nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    """
    Implements positional encoding
    """

    def aux(self, i, j):
        """
        auxiliary function that computes (i / 10000^(2j/d))
        """
        return i / 10000**(2*j/self.embedding_dim)

    def __init__(self, embedding_dim=32, max_len=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.P = torch.zeros((1, max_len, embedding_dim))
        # tensor tmp_aux of shape (max_len/2, embedding_dim)
        # tmp_aux[i, j] contains (i / 10000^(2j/d))
        tmp_aux = torch.tensor([
            [self.aux(i, j) for j in range(self.embedding_dim//2)]
            for i in range(max_len)
        ])
        self.P[:, :, 0::2] = torch.sin(tmp_aux)
        self.P[:, :, 1::2] = torch.cos(tmp_aux)

    def forward(self, x):
        """
        returns x + P where

        - x contains the embedding of each word of each sequence in the the batch
        - P is the positional encoding of each word of any sequence
        - P of shape (1, max_len, embedding_dim)
        - x of shape (N, max_len, embedding_dim)
        """
        out = x + self.P[:, :x.shape[1], :].to(x.device)
        return out

class DotProductAttention(nn.Module):

    def __init__(self, p_qk, p_v, embedding_dim, max_len=1000):
        # one FC layer for the keys, one for the values, and one for the query
        # Each fc layer will learn how to embed the values, queries, and keys
        # We don't use biases, we just want to learn 3 matrices W_q, W_k, W_k
        # with W_q and W_k of shape (max_len, p_qk)
        # and W_k of shape (max_len, p_v)
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.p_qk = p_qk
        self.p_v = p_v
        self.positionalEncoder = PositionalEncoder(embedding_dim, max_len)
        # Note that we use linear layers on multidimensional inputs
        # input of shape   (N, max_len, embedding dim)
        # weights of shape (p_qk, embedding_dim)
        # output of shape  (N, max_len, p_qk)
        self.fc_q = nn.Linear(embedding_dim, p_qk, bias=False)
        self.fc_k = nn.Linear(embedding_dim, p_qk, bias=False)
        # weights of shape (p_v, embedding_dim)
        # output of shape  (N, max_len, p_v)
        self.fc_v = nn.Linear(embedding_dim, p_v, bias=False)

    def _attention_pooling(self, Q, K, V, actual_lens=None):
        """
        Compute masked_softmax( QK^T / sqrt(d)) * V
        """
        # attention score of shape (N, max_len, max_len)
        d = self.p_qk
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)

        # attention weights of shape (N, max_len, max_len)
        #attention_weights = MaskedSoftmax(scores, actual_lens)
        attention_weights = F.softmax(scores, dim=-1)

        # output of shape (N, max_len, p_v)
        out = torch.bmm(attention_weights, V)
        return attention_weights, out

    def forward(self, x, actual_lens=None, positional_encoder=False):
        """
        x is of shape (N, max_len, embedding dim)
        actual_lens is of shape (N,)
        """
        # -------------------- Add positional encoding ------------------------
        # Add the positional encoding information to x
        if positional_encoder:
            x = self.positionalEncoder(x)

        # -------------- Defining queries, keys and values --------------------
        # Compute queries, keys and values from x
        # queries and keys of shape (N, max_len, p_qk)
        Q = self.fc_q(x)
        K = self.fc_k(x)
        # queries and keys of shape (N, max_len, p_v)
        V = self.fc_v(x)

        # ---------------------- Attention pooling ----------------------------
        # attention weights of shape (N, max_len, max_len)
        # pool of shape (N, max_len, p_v)
        alpha, pool = self._attention_pooling(Q, K, V, actual_lens)
        self.attention_weights = alpha
        return pool

class MultiHeadAttention(nn.Module):
    """
    Simple multihead dot product attention

    This simple implementatio is non-causal, with fixed sequence length,
    without parallel computing of heads and using positional encoding.
    """

    def __init__(self, h, p, embedding_dim, max_len=1000):
        # one FC layer for the keys, one for the values, and one for the query
        # Each fc layer will learn how to embed the values, queries, and keys
        # We don't use biases, we just want to learn 3 matrices W_q, W_k, W_k
        # with W_q and W_k of shape (max_len, p_qk)
        # and W_k of shape (max_len, p_v)
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        # Number of attention blocks
        self.h = h
        # Dimensions of queries, keys, values
        self.p = p

        # List of modules, each module being an attention block
        self.attention_blocks = nn.ModuleList([
            DotProductAttention(
                self.p, self.p, self.embedding_dim, self.max_len
            )
            for _ in range(self.h)
        ])
        self.positionalEncoder = PositionalEncoder(embedding_dim, max_len)
        self.fc_W_o = nn.Linear(self.h*self.p, embedding_dim, bias=False)

    def forward(self, x):
        """
        x is of shape (N, actual_len, embedding dim)
        actual_lens is of shape (N,)
        """
        # -------------------- Add positional encoding ------------------------
        # Add the positional encoding information to x
        # x is of shape (N, actual_len, embedding dim)
        x = self.positionalEncoder(x)
        # x is still of shape (N, actual_len, embedding dim)
        N = len(x)
        actual_len = x.shape[1]

        # -------------- Defining queries, keys and values --------------------
        # Get the output of each attention block
        heads = torch.zeros((self.h, N, actual_len, self.p)).to(device=x.device)
        for i in range(self.h):
            heads[i] = self.attention_blocks[i](x, positional_encoder=False)

        # Concatenate these outputs
        # heads of shape (actual_len, h, p)
        heads = heads.permute(1, 2, 0, 3)
        # heads of shape (N, actual_len, h*p)
        heads = torch.reshape(heads, (N, actual_len, -1))

        # Feed them to fc_Wo
        out = self.fc_W_o(heads)
        # out of shape (N, actual_len, embedding dim)
        return out

