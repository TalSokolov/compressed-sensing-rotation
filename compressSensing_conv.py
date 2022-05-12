import torch
import torch.nn.functional as F
import skimage.io as io
import os
import numpy as np
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
        y = torch.from_numpy(y).to(device).unsqueeze(0)
    return y


def load_w(mat_path='conf/default/mat.txt'):
    A = np.array([[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]])
    A = torch.from_numpy(A).float().unsqueeze(-1).unsqueeze(-1).to(device)
    return A


def save_ch(ch, channel_name):
    io.imsave(fname=os.path.join(tools.PROJ_PATH, '{}-{}.tiff'.format(channel_name, 'Tal')), arr=ch)
    return


def run(lambda_sparsity):

    lr = 0.1
    n_iter = 1500

    channels_names = tools.load_channels_names()
    y = load_y(tools.PROJ_PATH, tools.MULTI)
    gt = load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = load_w()

    x = torch.rand(1, 7, y.shape[-2], y.shape[-1]).to(device)
    x = torch.nn.Parameter(x, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)

    loss_list = []
    loss_recon_list = []
    loss_sparsity_list = []

    for i in range(n_iter):
        optimizer.zero_grad()
        # y_recon = F.conv2d(F.relu(x * m), w)
        y_recon = F.conv2d(F.relu(x), w)
        # loss_sparsity = torch.mean(torch.abs(x))
        loss_sparsity = (torch.count_nonzero(y) - torch.count_nonzero(y_recon))/(2024*2024*3)
        loss_recon = F.mse_loss(y_recon, y)
        loss = loss_recon + lambda_sparsity * loss_sparsity
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        loss_recon_list.append(loss_recon.item())
        loss_sparsity_list.append(loss_sparsity.item())

        if i % 100 == 0:
            print(f'Iteration {i}: loss={loss.item():.4f} | '
                  f'sparsity={loss_sparsity.item():.4f} | recon={loss_recon.item():.4f}')

        if i % 1000 == 0:

            for j, channel in enumerate(channels_names):
                io.imsave(os.path.join(tools.save_path, 'post_ref_{}.tif'.format(channel)),
                          F.relu(x)[0][j].detach().cpu().numpy(),
                          check_contrast=False)
    plt.figure()
    plt.plot(range(len(loss_list)), loss_list)
    plt.plot(range(len(loss_recon_list)), loss_recon_list)
    plt.plot(range(len(loss_sparsity_list)), loss_sparsity_list)
    plt.legend(['loss', 'recon', 'sparcity'])
    plt.savefig(os.path.join(tools.save_path, 'losses_new_sparsity_loss_{}.jpg'.format(lambda_sparsity)))
    plt.xlabel('iterations')
    plt.show()

    for j, channel in enumerate(channels_names):
        ch = F.relu(x)[0][j].detach().cpu().numpy()
        tools.evaluate(ch, gt[j], j)









