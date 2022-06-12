import torchvision.transforms as T
import torchvision.transforms.functional as TF
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random


def crop(y, crop_size):
    return T.RandomCrop([crop_size, crop_size])(y)


def augment(y):

    out = TF.rotate(img=y, angle=90 * random.randint(1, 4))
    out = T.RandomHorizontalFlip(0.5)(out)

    #i = random.uniform(0, 1)
    #if i < 0.3:
    #    return T.RandomHorizontalFlip(1)(y)

    #elif i < 0.5:
    #    return T.RandomVerticalFlip(1)(y)

    #else:
    #    return TF.rotate(img=y, angle=90*random.randint(1, 4))

    return out


def old_augment(x, y):

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


def plot_hetmap(file):
    res = pd.read_csv(file, index_col=0, sep='\t')
    fig = plt.figure()
    sns.heatmap(res, annot=True)
    fig.savefig('test',bbox_inches='tight')
