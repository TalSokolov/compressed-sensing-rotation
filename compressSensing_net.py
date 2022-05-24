import torch
import torch.nn.functional as F
from itertools import chain
import skimage.io as io
import os
import tools
from DIP.models.skip import skip
import argparse
from datetime import datetime
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
parser = argparse.ArgumentParser(description='Compressed Sensing')
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--n_iter', type=int, default=100000)
parser.add_argument('--input_dim', type=int, default=32)
parser.add_argument('--lambda_sparsity', type=float, default=1)
parser.add_argument('--noise', type=float, default=0)


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


def opt(w, y, gt, lambda_sparsity, channels_names, save_path='outputs', lr=0.005, n_iter=100000, input_dim=32, rand_noise=0):
    run_name = 'lr {} input dim {} sparcity loss {} noise {}'.format(lr, input_dim, lambda_sparsity, rand_noise)
    wandb.init(project="CSR", entity="talso", name=run_name)
    wandb.config = {
        "learning_rate": lr,
        "epochs": n_iter,
        "input_dim": input_dim,
        "sparcity_loss": lambda_sparsity,
        "noise": rand_noise
    }
    now = datetime.now()
    time = now.strftime("%m%d_%H%M")
    net = create_net(input_dim)
    noise = torch.randn(1, input_dim, y.shape[-2], y.shape[-1]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print("start running with {}".format(run_name))

    loss_list = []
    loss_recon_list = []
    loss_sparsity_list = []

    for i in range(n_iter):
        optimizer.zero_grad()
        net_input = noise + (noise.normal_() * rand_noise)
        x = net(net_input)
        y_recon = F.conv2d(x, w)
        loss_sparsity = (torch.count_nonzero(y) - torch.count_nonzero(y_recon))/(2024*2024*3) #torch.mean(torch.abs(x))#
        loss_recon = F.mse_loss(y_recon, y)
        loss = loss_recon + lambda_sparsity * loss_sparsity
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        loss_recon_list.append(loss_recon.item())
        loss_sparsity_list.append(loss_sparsity.item())
        wandb.log({"loss": loss})
        wandb.log({"loss reconstructin": loss_recon})
        wandb.log({"loss sparcity": loss_sparsity})

        if i % 10 == 0:
            print(f'Iteration {i}: loss={loss.item():.4f} | '
                  f'sparsity={loss_sparsity.item():.4f} | recon={loss_recon.item():.4f}')
            tools.plot_losses([loss_list, loss_recon_list, loss_sparsity_list], run_name)


        if i % 100 == 0:
            for j, channel in enumerate(channels_names):
                io.imsave(os.path.join(save_path, 'pred_{}_{}.tif'.format(channel, run_name)),

                          x[0][j].detach().cpu().numpy(),
                          check_contrast=False)

        for j, channel in enumerate(channels_names):
            ch = F.relu(x)[0][j].detach().cpu().numpy()
            tools.evaluate(ch, gt[j], j, lambda_sparsity)


    return noise


def run(args):
    channels_names = tools.load_channels_names()
    y = tools.load_y(tools.PROJ_PATH, tools.MULTI)
    gt = tools.load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = tools.load_w()
    opt(w, y, gt, lambda_sparsity=args.lambda_sparsity, channels_names=channels_names,
        save_path=tools.save_path.split('compressed-sensing-rotation/')[-1],
        input_dim=args.input_dim, n_iter=args.n_iter, lr=args.lr, rand_noise=args.noise)
    #

if __name__ == '__main__':

    args = parser.parse_args()

    run(args)
