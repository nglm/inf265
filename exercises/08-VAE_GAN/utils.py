
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms
from datetime import datetime
from typing import Sequence
from matplotlib import pyplot as plt

def load_MNIST(train_val_split=0.9, data_path='../data/', preprocessor=None, label_list=[]):

    # Define preprocessor if not already given
    if preprocessor is None:
        preprocessor = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(20),
        ])

    # load datasets
    data_train_val = datasets.MNIST(
        data_path,
        train=True,
        download=True,
        transform=preprocessor)

    data_test = datasets.MNIST(
        data_path,
        train=False,
        download=True,
        transform=preprocessor)

    # train/validation split
    n_train = int(len(data_train_val)*train_val_split)
    n_val =  len(data_train_val) - n_train

    # Add seed so that we get the same dataloaders
    data_train, data_val = random_split(
        data_train_val,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(123)
    )

    # Now define a lighter version of the dataset, keeping only some labels
    if label_list:
        label_map = {label:i for i, label in enumerate(label_list)}

        mnist_train = [(img, label_map[label]) for img, label in data_train if label in label_list]
        mnist_val = [(img, label_map[label]) for img, label in data_val if label in label_list]
        mnist_test = [(img, label_map[label]) for img, label in data_test if label in label_list]
    else:
        mnist_train = data_train
        mnist_val = data_val
        mnist_test = data_test

    print('Size of the training dataset: ', len(mnist_train))
    print('Size of the validation dataset: ', len(mnist_val))
    print('Size of the test dataset: ', len(mnist_test))

    return (mnist_train, mnist_val, mnist_test)


class LeNet5(nn.Module):
    """
    Regular image classifier
    - input: image
    - output: label predicted
    """

    def __init__(self, n_labels=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=6, kernel_size=5, stride=1)
        self.fc3 = nn.Linear(in_features=864, out_features=120)
        self.fc4 = nn.Linear(in_features=120, out_features=n_labels)

    def forward(self, x):
        N = x.shape[0]
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(N, -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

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


def plot_true_VS_reconstructed(ae, imgs):
    """
    Plot side by side original images with their reconstructed counterpart using a trained AE
    """
    ae.eval()
    N_img = 25
    fig, axs = plt.subplots(nrows=5, ncols=10, figsize=(10,6), sharex=True, sharey=True)
    for i, img in enumerate(imgs[:N_img]):
        with torch.no_grad():
            out = ae(img.unsqueeze(0))
            # True image
            axs.flat[2*i].imshow(img.permute(1, 2, 0), cmap='Greys')
            # Reconstruction
            axs.flat[2*i + 1].imshow(out.squeeze(0).permute(1, 2, 0), cmap='Greys')
            # Set ax title for the first row
            if i<5:
                axs.flat[2*i].set_title("True\nimage")
                axs.flat[2*i + 1].set_title("AE recon-\nstruction")
    return fig, axs

def relative_error(a, b):
    return (torch.norm(a - b) / torch.norm(a))

def int_to_pair(n):
    """
    Return `(n, n)` if `n` is a int or `n` if `n` is already a tuple of length 2
    """
    # If n is a float or integer
    if not isinstance(n, Sequence):
        return (int(n), int(n))
    elif len(n) == 1:
        return (int(n[0]), int(n[0]))
    elif len(n) == 2:
        return ( int(n[0]), int(n[1]) )
    else:
        raise ValueError("Please give an int or a pair of int")

def initialize_weights(C_in, C_out, kernel_size):
    """
    Helper function for the tests in 'test_transpose_conv'
    """
    kernel_size = int_to_pair(kernel_size)
    len_weights = C_in*C_out*kernel_size[0]*kernel_size[1]
    weights = (torch.arange(len_weights) - len_weights/2)/10
    return weights.reshape(C_in, C_out, kernel_size[0], kernel_size[1])

def test_transpose_conv(apply_transpose_conv):
    out1_exp = torch.Tensor([[[[ -30.,  -55.,  -55.,  -25.],
        [ -50.,  -90.,  -90.,  -40.],
        [ -60., -105., -105.,  -45.],
        [ -60., -105., -105.,  -45.],
        [ -30.,  -50.,  -50.,  -20.],
        [ -10.,  -15.,  -15.,   -5.]],

        [[   0.,    5.,    5.,    5.],
        [  10.,   30.,   30.,   20.],
        [  30.,   75.,   75.,   45.],
        [  30.,   75.,   75.,   45.],
        [  30.,   70.,   70.,   40.],
        [  20.,   45.,   45.,   25.]]],


        [[[ -30.,  -55.,  -55.,  -25.],
        [ -50.,  -90.,  -90.,  -40.],
        [ -60., -105., -105.,  -45.],
        [ -60., -105., -105.,  -45.],
        [ -30.,  -50.,  -50.,  -20.],
        [ -10.,  -15.,  -15.,   -5.]],

        [[   0.,    5.,    5.,    5.],
        [  10.,   30.,   30.,   20.],
        [  30.,   75.,   75.,   45.],
        [  30.,   75.,   75.,   45.],
        [  30.,   70.,   70.,   40.],
        [  20.,   45.,   45.,   25.]]]])

    out2_exp = torch.Tensor([[[[ 1.87199997e+02,  3.45299988e+02,  3.44749969e+02,  1.57900024e+02],
        [ 3.15900024e+02,  5.73850037e+02,  5.72950012e+02,  2.57650024e+02],
        [ 3.86400024e+02,  6.86250122e+02,  6.85200073e+02,  2.99550049e+02],
        [ 3.84599976e+02,  6.83099976e+02,  6.82049988e+02,  2.98200073e+02],
        [ 1.99200012e+02,  3.41249969e+02,  3.40749939e+02,  1.41950043e+02],
        [ 7.10999832e+01,  1.13749985e+02,  1.13599983e+02,  4.26499939e+01]],

        [[ 1.43999786e+01,  1.52587891e-05,  5.00183105e-02, -1.43000031e+01],
        [-2.87999954e+01, -1.14950035e+02, -1.14650032e+02, -8.58500290e+01],
        [-1.29299973e+02, -3.44249939e+02, -3.43500000e+02, -2.14350006e+02],
        [-1.28399994e+02, -3.42000000e+02, -3.41250000e+02, -2.13000031e+02],
        [-1.41899979e+02, -3.40349976e+02, -3.39650024e+02, -1.97949997e+02],
        [-9.90000000e+01, -2.26150009e+02, -2.25700012e+02, -1.26849998e+02]]],


        [[[ 1.69199982e+02,  3.12300018e+02,  3.11750061e+02,  1.42900024e+02],
        [ 2.85900024e+02,  5.19850098e+02,  5.18950012e+02,  2.33649994e+02],
        [ 3.50400024e+02,  6.23250000e+02,  6.22200012e+02,  2.72549988e+02],
        [ 3.48599945e+02,  6.20099854e+02,  6.19049927e+02,  2.71200012e+02],
        [ 1.81199982e+02,  3.11250031e+02,  3.10750000e+02,  1.29949997e+02],
        [ 6.50999985e+01,  1.04750008e+02,  1.04599998e+02,  3.96500015e+01]],

        [[ 1.44000015e+01,  3.00000763e+00,  3.05000305e+00, -1.12999954e+01],
        [-2.27999954e+01, -9.69499969e+01, -9.66499939e+01, -7.38500214e+01],
        [-1.11299980e+02, -2.99249939e+02, -2.98500031e+02, -1.87350037e+02],
        [-1.10399994e+02, -2.97000000e+02, -2.96250000e+02, -1.86000031e+02],
        [-1.23899971e+02, -2.98350037e+02, -2.97650024e+02, -1.73950012e+02],
        [-8.69999924e+01, -1.99150024e+02, -1.98700012e+02, -1.11850006e+02]]]])

    N = 2
    H_in = 4
    W_in = 3
    C_in = 5
    C_out = 2

    kernel_size = (3,2)
    stride = 1

    weights = initialize_weights(C_in, C_out, kernel_size)
    x1 = torch.ones((N, C_in, H_in, W_in))*10
    x2 = torch.arange(120).reshape(N, C_in, H_in, W_in)/10 - 60

    print("weights shape:     ", weights.shape)
    print("input shape:       ", x1.shape)
    print("shape expected:    ", out1_exp.shape)

    out1 = apply_transpose_conv(x1, weights, stride)
    out2 = apply_transpose_conv(x2, weights, stride)

    print("out1.shape:        ", out1.shape)
    print("out2.shape:        ", out2.shape)

    print("Relative error x1:  {:.5f}".format(relative_error(out1, out1_exp)))
    print("Relative error x2:  {:.5f}".format(relative_error(out2, out2_exp)))

