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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
parser = argparse.ArgumentParser(description='Compressed Sensing')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--n_iter', type=int, default=10000)
parser.add_argument('--lambda_sparsity', type=float, default=1)
parser.add_argument('--lambda_mask', type=float, default=1)
parser.add_argument('--noise', type=float, default=0)
parser.add_argument('--log', type=bool, default=True)


def create_net():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.net = skip(
                3, 7,
                num_channels_down=[8, 16, 32, 64, 128],
                num_channels_up=[8, 16, 32, 64, 128],
                num_channels_skip=[4, 4, 4, 4, 4],
                upsample_mode='bilinear', filter_size_down=5, filter_size_up=5,
                need_relu=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').to(device)

        def forward(self, net_input):
            return self.net(net_input)

    return Net()


def opt(w, y, gt, other_ys, lambda_sparsity, channels_names, lr, n_iter,
        rand_noise, log, save_path='outputs'):

    # logging:
    run_name = 'lr {} sparsity loss {} noise {} n_iter {}'.format(lr, lambda_sparsity, rand_noise, n_iter)

    if log:
        wandb.init(project="CSR", entity="talso", name=run_name)
        wandb.config = {
            "learning_rate": lr,
            "epochs": n_iter,
            "sparcity_loss": lambda_sparsity,
            "noise": rand_noise
        }

    # net
    net = create_net()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print("start running with {}".format(run_name))

    for i in range(n_iter):
        optimizer.zero_grad()
        other = False
        iter_input = y

        if random.uniform(0, 1) > 0.5:
            iter_input = other_ys[random.randint(0, len(other_ys) - 1)]
            other = True

        if random.uniform(0, 1) > 0.5:
            iter_input = augmentations.augment(iter_input)
            other = True

        x = net(iter_input)
        y_recon = F.conv2d(x, w)
        loss_sparsity = torch.mean(torch.abs(x))
        loss_recon = F.mse_loss(y_recon, iter_input)
        loss = loss_recon + lambda_sparsity * loss_sparsity
        loss.backward()
        optimizer.step()
        if log and not other:
            wandb.log({"loss": loss})
            wandb.log({"loss reconstructin": loss_recon})
            wandb.log({"loss sparcity": loss_sparsity})

        if i % 100 == 0:
            print(f'Iteration {i}: loss={loss.item():.4f} | '
                  f'sparsity={loss_sparsity.item():.4f} | recon={loss_recon.item():.4f}')

    full_x = net(y)

    for j, channel in enumerate(channels_names):
        ch = F.relu(full_x)[0][j].detach().cpu().numpy()
        tools.evaluate(ch, gt[j], j, run_name)

        io.imsave(os.path.join(save_path, 'pred_{}_{}.tif'.format(channel, run_name)),
                  full_x[0][j].detach().cpu().numpy(),
                  check_contrast=False)

    return


def run(args):
    channels_names = tools.load_channels_names()
    y = tools.load_y(tools.PROJ_PATH, tools.MULTI)
    other_ys = [tools.load_y(tools.OTHERS_PATH, ['{} {}'.format(ch, str(i)) for ch in tools.MULTI]) for i in range(1, 9)]
    gt = tools.load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = tools.load_w()
    opt(w, y, gt, other_ys, lambda_sparsity=args.lambda_sparsity, channels_names=channels_names,
        n_iter=args.n_iter, lr=args.lr, rand_noise=args.noise, log=args.log,
        save_path=tools.save_path.split('compressed-sensing-rotation/')[-1])


if __name__ == '__main__':

    args = parser.parse_args()

    run(args)
