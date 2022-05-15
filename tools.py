import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import skimage.io as io

CHANNELS = ['CD45', 'CD8', 'dsDNA', 'Ki67', 'Pan-Keratin', 'HLA-DR', 'SMA']
MULTI = ['144_HLA_Keratin_Ki67_SMA', '147_CD8_HLA_dsDNA_SMA', '153_SMA_dsDNA_Keratin_CD45']
PROJ_PATH = 'datasets/multi_clean_1304/B428_C'
save_path = 'iterations/U_net'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_channels_names():
    channels = CHANNELS
    return channels


def evaluate(ch, gt, idx):

    bool_ch = (np.matrix(ch) > 0).astype(int)
    bool_gt = (np.matrix(gt) > 0).astype(int)
    FP = np.sum((bool_ch - bool_gt) > 0)/(1024**2)
    FN = np.sum((bool_gt - bool_ch) > 0)/(1024**2)
    TP = np.sum((bool_gt + bool_ch) == 2)/(1024**2)
    TN = np.sum((bool_gt + bool_ch) == 0)/(1024**2)

    plt.figure(idx)
    plt.pie([FP, FN, TP, TN], labels=['False Positive', 'False Negative', 'True Positive', 'True Negative'], autopct='%1.1f%%')
    plt.title('Channel {}'.format(CHANNELS[idx]))
    plt.savefig(os.path.join(PROJ_PATH, 'eval {}.tiff'.format(CHANNELS[idx])))

    return


def plot_losses(loss, lr, imput_dim):

    loss_list = loss[0]
    loss_recon_list = loss[1]
    loss_sparsity_list = loss[2]

    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.plot(range(len(loss_recon_list)), loss_recon_list)
    plt.plot(range(len(loss_sparsity_list)), loss_sparsity_list)
    plt.legend(['loss', 'recon', 'sparcity'])
    plt.savefig(os.path.join(save_path, 'losses_lr_{}_inputdim_{}.jpg'.format(lr, imput_dim)))
    plt.xlabel('iterations')
    plt.show()



def load_y(path, ch, stack=True):
    y = []
    for channel in ch:
        try:
            v = io.imread(os.path.join(path, '{}.tif'.format(channel)))
        except FileNotFoundError:
            v = io.imread(os.path.join(path, '{}.tiff'.format(channel)))
        y.append(v)
    if stack:
        y = np.stack(y).astype(np.float32)
        y = torch.from_numpy(y).to(device).unsqueeze(0)
    return y


def load_w(mat_path='conf/default/mat.txt', basic=False):
    A = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]])
    if basic:
        A = torch.from_numpy(A).float().to(device)
    else:
        A = torch.from_numpy(A).float().unsqueeze(-1).unsqueeze(-1).to(device)

    return A

def save_ch(ch, channel_name):
    io.imsave(fname=os.path.join(PROJ_PATH, '{}-{}.tiff'.format(channel_name, 'Tal')), arr=ch)
    return
