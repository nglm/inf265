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

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
import numpy as np
from datetime import datetime

def set_device(device=None):
    """
    Helper function to set device
    """
    if device is None:
        device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
        print(f"On device {device}.")
    return device

def train(n_epochs, optimizer, model, loss_fn, train_loader, device=None):

    device = set_device(device)

    n_batch = len(train_loader)
    losses_train = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, n_epochs + 1):

        loss_train = 0.0
        for contexts, targets in train_loader:

            contexts = contexts.to(device=device)
            targets = targets.to(device=device)

            outputs = model(contexts)

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_train += loss.item()

        losses_train.append(loss_train / n_batch)

        if epoch == 1 or epoch % 5 == 0:
            print('{}  |  Epoch {}  |  Training loss {:.5f}'.format(
                datetime.now().time(), epoch, loss_train / n_batch))
    return losses_train


def compute_accuracy(model, loader, device=None):
    model.eval()
    device = set_device(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for contexts, targets in loader:
            contexts = contexts.to(device=device)
            targets = targets.to(device=device)

            outputs = model(contexts)
            _, predicted = torch.max(outputs, dim=1)
            total += len(targets)
            correct += int((predicted == targets).sum())

    acc =  correct / total
    return acc

def model_selection(
    model_class, model_params, optimizers, optim_params, n_epochs, loss_fn,
    train_loader, val_loader,
    seed=265, model_name='model_', device=None,
):

    device = set_device(device)

    accuracies = []
    models = []

    for i in range(len(optim_params)):

        print("   Current parameters: ")
        print("".join(['%s = %s\n' % (key, value) for (key, value) in optim_params[i].items()]))

        torch.manual_seed(seed)
        model = model_class(*model_params)
        model.to(device=device)

        optimizer = optimizers[i](
            model.parameters(),
            **optim_params[i]
        )

        train(
            n_epochs = n_epochs,
            optimizer = optimizer,
            model = model,
            loss_fn = loss_fn,
            train_loader = train_loader,
        )

        acc_train = compute_accuracy(model, train_loader,device=device)
        acc_val = compute_accuracy(model, val_loader, device=device)
        print("Training Accuracy:     %.4f" %acc_train)
        print("Validation Accuracy:   %.4f" %acc_val)

        accuracies.append(acc_val)
        models.append(model)

        name = model_name + "_".join(['%s=%s' %(k, v) for (k, v) in optim_params[i].items()]) + '.pt'
        torch.save(model, name)

    i_best_model = np.argmax(accuracies)
    best_model = models[i_best_model]
    torch.save(best_model, model_name[:-1]+'.pt')
    return best_model, i_best_model

def model_evaluation(model, train_loader, val_loader, test_loader, device=None):
    acc_train = compute_accuracy(model, train_loader,device=device)
    acc_val = compute_accuracy(model, val_loader, device=device)
    acc_test = compute_accuracy(model, test_loader, device=device)
    print("Training Accuracy:     %.4f" %acc_train)
    print("Validation Accuracy:   %.4f" %acc_val)
    print("Test Accuracy:         %.4f" %acc_test)
    return acc_test


