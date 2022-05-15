import torch
import numpy as np
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import tools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run():
    channels_names = tools.load_channels_names()
    y = tools.load_y(tools.PROJ_PATH, tools.MULTI)
    gt = tools.load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = tools.load_w(basic=True)

    x = np.array([[nnls(w, y[i][j])[0] for i in range(y.shape[1])] for j in range(y.shape[0])])

    for i in range(7):
        ch = np.split(x, 7, 2)[i].squeeze()
        ch = np.rot90(ch, k=1, axes=(0, 1))
        ch = np.flip(ch, axis=0)
        tools.save_ch(ch, channels_names[i])
        tools.evaluate(ch, gt[i], i)

    plt.show()

    y_hat = np.array([[np.array(w)@x[i][j] for i in range(x.shape[1])] for j in range(x.shape[0])])
    print(torch.sum(y-y_hat > 10**-16)/((1024**2)*3))


if __name__ == '__main__':

    run()
