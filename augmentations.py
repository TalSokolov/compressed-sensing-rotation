import tools
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import skimage.io as io
import numpy as np
import seaborn as sns
import pandas as pd
import PyQt5
import matplotlib
import matplotlib.pyplot as plt
import random

def augment(x, y):

    i = random.uniform(0, 1)
    if i < 0.3:
        R, L, H, W = T.RandomCrop.get_params(y, output_size=[512, 512])
        aug_x = TF.crop(x, R, L, H, W)
        aug_y = TF.crop(y, R, L, H, W)

    elif i < 0.5:
        aug_x = T.RandomHorizontalFlip(1)(x)
        aug_y = T.RandomHorizontalFlip(1)(y)

    elif i < 0.7:
        aug_x = T.RandomVerticalFlip(1)(x)
        aug_y = T.RandomVerticalFlip(1)(y)

    else:
        a = T.RandomRotation.get_params(degrees=(-180, 180))
        aug_x = TF.rotate(x, a)
        aug_y = TF.rotate(y, a)

    return [aug_x, aug_y]

#def flip_horizontal(y):
    #y = y.numpy()
#    io.imsave(tools.PROJ_PATH + '/test_in.tiff', arr=y.detach().cpu().numpy())

    #io.imsave(tools.PROJ_PATH + '/test_out.tiff', arr=np.stack(T.RandomHorizontalFlip(1)(y)))
    #io.imsave(tools.PROJ_PATH + '/test_out.tiff', arr=np.stack(T.RandomVerticalFlip(1)(y)))
 #   r, l, h, w = T.RandomCrop.get_params(y, output_size=[512, 512])

#    io.imsave(tools.PROJ_PATH + '/test_out1.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
#    io.imsave(tools.PROJ_PATH + '/test_out2.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
#    io.imsave(tools.PROJ_PATH + '/test_out3.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
#    io.imsave(tools.PROJ_PATH + '/test_out4.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
#    io.imsave(tools.PROJ_PATH + '/test_out5.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
#    r, l, h, w = T.RandomCrop.get_params(y, output_size=[512, 512])
#    io.imsave(tools.PROJ_PATH + '/test_out6.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))


    #io.imsave(tools.PROJ_PATH + '/test_out.tiff', arr=y.detach().cpu().numpy())


def plot_hetmap(file):
    res = pd.read_csv(file, index_col=0)
    fig = plt.figure()
    sns.heatmap(res, annot=True)
    fig.savefig('test', pad_inches=1)


if __name__ == '__main__':

    y = tools.load_y(tools.PROJ_PATH, tools.MULTI)
    augment(y, y)