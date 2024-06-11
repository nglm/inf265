import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset

def create_dataset(
    text, vocab, context_size,
    black_list=["<unk>", ",", ".", "!", "?", '"'], white_list=None,
    occ_max=np.inf, map_target=None, bidirectional=True, use_unk_limit=True,
):
    contexts = []
    targets = []

    n_text = len(text)
    n_vocab = len(vocab)
    exceptions = [vocab[e] for e in black_list]

    if white_list is None:
        white_list = list(range(n_vocab))
    else:
        white_list = [vocab[w] for w in white_list]

    # Change labels if only a few target are kept (be and have conjugation)
    if map_target is None:
        map_target = {i:i for i in range(n_vocab)}
    target_list = [t for t in white_list if t not in exceptions]

    # Transform the text as a list of labels.
    txt = [vocab[w] for w in text]

    # To limit imbalance
    counts = [0]*n_vocab

    if use_unk_limit:
        if bidirectional:
            unk_limit = context_size
        else:
            unk_limit = context_size / 2
    else:
        unk_limit = np.inf

    for i in range(n_text - (context_size*2)):
        # target
        t = txt[i + context_size]
        if t in target_list and counts[t] < occ_max:

            # Context before
            c = txt[i:i + context_size]
            # Context after
            if bidirectional:
                c += txt[i + context_size + 1:i + 2*context_size + 1]

            # Ignore this context / target if there are too many unknown
            if c.count(0) > unk_limit:
                pass
            else:
                counts[t] += 1
                targets.append(map_target[t])
                contexts.append(torch.tensor(c))

    contexts = torch.stack(contexts)
    targets = torch.tensor(targets)
    return TensorDataset(contexts, targets)

class CBoW(nn.Module):

    def __init__(self, vocab_size, embedding_dim=16, context_size=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*context_size*2, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_size = context_size

    def forward(self, x, predict=True):
        self.emb = self.embeddings(x)
        if predict:
            out = F.relu(torch.flatten(self.emb, 1))
            out = self.fc1(out)
        else:
            out = self.emb
        return out
