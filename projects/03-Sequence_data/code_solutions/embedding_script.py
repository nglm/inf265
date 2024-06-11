# %%
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

from utils import model_selection, model_evaluation, set_device
from cbow import create_dataset, CBoW
from embedding_utils import similarity_matrix, find_N_closest

seed = 265
torch.manual_seed(seed)
device = set_device()

# %%
# List of words contained in the dataset
generated_path = '../generated/'
list_words_train = torch.load(generated_path + 'books_train.pt')
list_words_val = torch.load(  generated_path + 'books_val.pt')
list_words_test = torch.load( generated_path + 'books_test.pt')

# vocab contains the vocabulary found in the data, associating an index to each word
vocab = torch.load( generated_path + 'vocab.pt')
weight = torch.load(generated_path + 'weight.pt')

vocab_size = len(vocab)

print("Total number of words in the dataset:   ", len(list_words_train))
print("Total number of words in the dataset:   ", len(list_words_val))
print("Number of distinct words kept:          ", vocab_size)

def pipeline(
    context_size, embedding_dim, occ_max=np.inf, use_weight=True, use_unk_limit=True,
    black_list=["<unk>", ",", ".", "!", "?", '"'],
    generated_path='../generated/'
):
    """
    Warning: this function relies heavily on global variables and default parameters
    """
    device = set_device()
    
    print("="*59)
    print(
        "Context size  %d  |  Embedding dim  %d  |  occ_max  %s  |  weights %s"
        %(context_size, embedding_dim, str(occ_max), str(use_weight) )
    )
    print(
        "use_unk_limit %s " %(str(use_unk_limit))
    )
    print("Black_list: %s" %" | ".join(black_list))

    # -------------- Datasets -------------
    data_train_ngram = create_dataset(list_words_train, vocab, context_size, black_list=black_list, occ_max=occ_max, use_unk_limit=use_unk_limit)
    data_val_ngram = create_dataset(list_words_val,     vocab, context_size, black_list=black_list, occ_max=occ_max, use_unk_limit=use_unk_limit)
    data_test_ngram = create_dataset(list_words_test,   vocab, context_size, black_list=black_list, occ_max=occ_max, use_unk_limit=use_unk_limit)

    print(len(data_train_ngram))
    print(len(data_val_ngram))
    print(len(data_test_ngram))

    train_loader = DataLoader(data_train_ngram, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val_ngram, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test_ngram, batch_size=batch_size, shuffle=True)

    # ------- Loss function parameters -------
    if use_weight:
        loss_fn = nn.CrossEntropyLoss(weight=weight.to(device=device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    # ---------- Optimizer parameters --------
    list_lr = [0.001]
    optimizers = [optim.Adam for _ in range(len(list_lr))]
    optim_params = [{
            "lr" : list_lr[i],
        } for i in range(len(list_lr))]

    # -------- Model class parameters --------
    model_class = CBoW
    model_params = (vocab_size, embedding_dim, context_size)
    
    # ----------- Model name -----------------
    model_name = generated_path +'CBoW_'
    hyperparams = {
        "context": context_size,
        "emb_dim": embedding_dim,
        "weights": use_weight,
        "unk_limit": use_unk_limit,
        "occ_max": occ_max, 
    }
    model_name += "_".join(['%s=%s' %(k, v) for (k, v) in hyperparams.items()]) + '.pt'

    # ----------- Model selection -----------
    model_cbow, i_best_model = model_selection(
        model_class, model_params, optimizers, optim_params,
        n_epochs, loss_fn,
        train_loader, val_loader,
        seed=265, model_name=model_name, device=device
    )

    # ----------- Model evaluation -----------
    test_acc = model_evaluation(model_cbow, train_loader, val_loader, test_loader, device=device)

    # ----------- Embedding analysis -----------
    sim, embs = similarity_matrix(vocab, model_cbow)
    words = [
        'the', 'table', "man", 'little', 'big', 'always', 'mind', 'black', 'white', 'child', 'children', 
        'yes', 'out', "me", "have", "be"
    ]
    for w in words:
        print('-'*59)
        find_N_closest(sim, w, vocab)
        
    return model_cbow, embs, sim

# %%
n_epochs = 30
batch_size = 1024

list_context_size = [2, 3, 4]
list_embedding_dim = [8, 12, 16]
list_occ_max = [10000, np.inf]
list_use_weight = [True, False]

for context_size in list_context_size:
    for embedding_dim in list_embedding_dim:
        for occ_max in list_occ_max:
            for use_weight in list_use_weight:
                pipeline(context_size, embedding_dim, occ_max, use_weight)

# %%



