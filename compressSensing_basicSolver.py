import torch
import skimage.io as io
import os
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import tools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_y(path, ch, stack=True):
    y = []
    for channel in ch:
        try:
            v = io.imread(os.path.join(path, '{}.tiff'.format(channel)))
        except FileNotFoundError:
            v = io.imread(os.path.join(path, '{}.tif'.format(channel)))
        y.append(v)
    if stack:
        y = np.stack(y).astype(np.float32)
        y = torch.from_numpy(y).to(device).permute(1, 2, 0)#.unsqueeze(0)
    return y


def load_w(mat_path='conf/default/mat.txt'):
    A = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]])
    A = torch.from_numpy(A).float().to(device)
    return A


def save_ch(ch, channel_name):
    io.imsave(fname=os.path.join(tools.PROJ_PATH, '{}-{}.tiff'.format(channel_name, 'Tal')), arr=ch)
    return


def run():
    channels_names = tools.load_channels_names()
    y = load_y(tools.PROJ_PATH, tools.MULTI)
    gt = load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = load_w()

    x = np.array([[nnls(w, y[i][j])[0] for i in range(y.shape[1])] for j in range(y.shape[0])])

    for i in range(7):
        ch = np.split(x, 7, 2)[i].squeeze()
        ch = np.rot90(ch, k=1, axes=(0, 1))
        ch = np.flip(ch, axis=0)
        save_ch(ch, channels_names[i])
        tools.evaluate(ch, gt[i], i)

    plt.show()

    y_hat = np.array([[np.array(w)@x[i][j] for i in range(x.shape[1])] for j in range(x.shape[0])])
    print(torch.sum(y-y_hat > 10**-16)/((1024**2)*3))







