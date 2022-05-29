import torch
import torch.nn.functional as F
import skimage.io as io
import os
import tools
from DIP.models.skip import skip
import argparse
import wandb
import random
import augmentations
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
parser = argparse.ArgumentParser(description='Compressed Sensing')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--input_dim', type=int, default=0)
parser.add_argument('--lambda_sparsity', type=float, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--IL', type=bool, default=False)
parser.add_argument('--log', type=bool, default=True)


def create_net(input_dim):
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = skip(
                input_dim, 7,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[4, 4, 4, 4, 4],
                upsample_mode='bilinear', filter_size_down=5, filter_size_up=5,
                need_relu=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

        def forward(self, net_input):
            return self.net(net_input)

    return Net()


def opt(w, y, gt, lambda_sparsity, channels_names, lr, n_iter, input_dim,
        rand_noise, IL, log, save_path='outputs'):

    # logging:
    run_name = 'lr {} input dim {} sparsity loss {} noise {} n_iter {} IL {}'.format(lr, input_dim, lambda_sparsity,
                                                                                     rand_noise, n_iter, IL)

    io.imsave(os.path.join(save_path, 'INPUT_{}.tif'.format(run_name)),
              y.detach().cpu().numpy(),
              check_contrast=False)
    if log:
        wandb.init(project="CSR", entity="talso", name=run_name)
        wandb.config = {
            "learning_rate": lr,
            "epochs": n_iter,
            "input_dim": input_dim,
            "sparcity_loss": lambda_sparsity,
            "noise": rand_noise
        }


    # net
    if input_dim:
        net_input = torch.randn(1, input_dim, y.shape[-2], y.shape[-1]).to(device)
    else:
        net_input = y

    net = create_net(net_input.shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print("start running with {}".format(run_name))

    for i in range(n_iter):
        optimizer.zero_grad()
        if IL and random.uniform(0, 1) > 0.5:
            [iter_input, y_ref] = augmentations.augment(net_input, y)
            io.imsave(tools.save_path + '/test_in1.tiff', arr=np.stack(iter_input))
            io.imsave(tools.save_path + '/test_out1.tiff', arr=np.stack(y_ref))
        else:
            iter_input = net_input
            y_ref = y
        x = net(iter_input)

        y_recon = F.conv2d(x, w)
        loss_sparsity = torch.mean(torch.abs(x))
        loss_recon = F.mse_loss(y_recon, y_ref)
        loss = loss_recon + lambda_sparsity * loss_sparsity
        loss.backward()
        optimizer.step()
        if log:
            wandb.log({"loss": loss})
            wandb.log({"loss reconstructin": loss_recon})
            wandb.log({"loss sparcity": loss_sparsity})

        if i % 10 == 0:
            print(f'Iteration {i}: loss={loss.item():.4f} | '
                  f'sparsity={loss_sparsity.item():.4f} | recon={loss_recon.item():.4f}')

        if i % 100 == 0:
            full_x = net(net_input)

            for j, channel in enumerate(channels_names):
                io.imsave(os.path.join(save_path, 'pred_{}_{}.tif'.format(channel, run_name)),
                          full_x[0][j].detach().cpu().numpy(),
                          check_contrast=False)

            for j, channel in enumerate(channels_names):
                ch = F.relu(full_x)[0][j].detach().cpu().numpy()
                tools.evaluate(ch, gt[j], j, run_name)

    return net_input


def run(args):
    channels_names = tools.load_channels_names()
    y = tools.load_y(tools.PROJ_PATH, tools.MULTI)
    gt = tools.load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = tools.load_w()
    opt(w, y, gt, lambda_sparsity=args.lambda_sparsity, channels_names=channels_names,
        input_dim=args.input_dim, n_iter=args.n_iter, lr=args.lr, rand_noise=args.noise, IL=args.IL, log=args.log,
        save_path=tools.save_path.split('compressed-sensing-rotation/')[-1])


if __name__ == '__main__':

    args = parser.parse_args()

    run(args)
