import tools
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch
import skimage.io as io
import numpy as np


def flip_horizontal(y):
    #y = y.numpy()
    io.imsave(tools.PROJ_PATH + '/test_in.tiff', arr=y.detach().cpu().numpy())

    #io.imsave(tools.PROJ_PATH + '/test_out.tiff', arr=np.stack(T.RandomHorizontalFlip(1)(y)))
    #io.imsave(tools.PROJ_PATH + '/test_out.tiff', arr=np.stack(T.RandomVerticalFlip(1)(y)))
    r, l, h, w = T.RandomCrop.get_params(y, output_size=[512, 512])

    io.imsave(tools.PROJ_PATH + '/test_out1.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
    io.imsave(tools.PROJ_PATH + '/test_out2.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
    io.imsave(tools.PROJ_PATH + '/test_out3.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
    io.imsave(tools.PROJ_PATH + '/test_out4.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
    io.imsave(tools.PROJ_PATH + '/test_out5.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))
    r, l, h, w = T.RandomCrop.get_params(y, output_size=[512, 512])
    io.imsave(tools.PROJ_PATH + '/test_out6.tiff', arr=np.stack(TF.crop(y, r, l, h, w)))


    #io.imsave(tools.PROJ_PATH + '/test_out.tiff', arr=y.detach().cpu().numpy())

if __name__ == '__main__':

    y = tools.load_y(tools.PROJ_PATH, tools.MULTI)
    flip_horizontal(y)
