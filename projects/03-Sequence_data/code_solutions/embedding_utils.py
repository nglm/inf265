import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from utils import set_device

def similarity_matrix(vocab, model, device=None, use_tanh=True):
    model.eval()
    vocab_size = len(vocab)
    sim = torch.zeros((vocab_size, vocab_size))

    device = set_device(device)

    idx = torch.arange(vocab_size).to(device)
    with torch.no_grad():
        embs = model(idx, predict=False)

        # Should we use tanh on the embeddings?
        if use_tanh:
            embs = torch.tanh(embs)
        embs = model(idx, predict=False)

    for i in range(vocab_size):
        idx = i*torch.ones(vocab_size, dtype=torch.int).to(device)
        with torch.no_grad():
            emb = model(idx, predict=False)
            if use_tanh:
                emb = torch.tanh(emb)
            emb = model(idx, predict=False)
            sim[i] = F.cosine_similarity(embs, emb)

    return sim, embs

def find_N_closest(sim, word, vocab, N=10, opposite=False, verbose=True):
    idx = vocab[word]
    sims = sim[idx]
    ind_sorted_sim = torch.argsort(sims, descending=not opposite)
    closest_words = vocab.lookup_tokens(list(ind_sorted_sim[:N]))
    closest_sims = [sims[i] for i in ind_sorted_sim[:N]]
    if verbose:
        print(word)
        for i, (w, s) in enumerate(zip(closest_words, closest_sims)):
            print("%d  |   similitude: %f   |   %s " %(i, s, w))
    return closest_words, closest_sims

