import torch
import torch.nn.functional as F
import skimage.io as io
import os
import tools
from DIP.models.skip import skip
from DIP.models.skip_omer import skip_omer
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
parser.add_argument('--crop_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--train_size', type=int, default=1)


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


def create_mask_net():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.netG = skip_omer().to(device)

        # no need to use this function at inference. just call netG, as in the previous message
        def forward(self, input):
            outputs = {}
            outputs['x_pred'] = self.netG(input['y'])

            return outputs

    return Model()


def calculate_mask_loss(output, mask):
    neg_mask = 1 - mask
    outputs_should_be_zero = output * neg_mask
    loss = (outputs_should_be_zero ** 2).mean()
    return loss


def opt(w, y, gt, other_ys, ys, lambda_sparsity, lambda_mask, channels_names, lr, n_iter,
        rand_noise, crop_size, log, batch_size, train_size, save_path='outputs'):

    # logging:
    run_name = 'lr {} sparsity loss {} mask loss {} noise {} n_iter {} crop {} batch size {} train size {}'\
        .format(lr, lambda_sparsity, lambda_mask, rand_noise, n_iter, crop_size, batch_size, train_size)

    if log:
        wandb.init(project="CSR_masks_fix", entity="talso", name=run_name)
        wandb.config = {
            "learning_rate": lr,
            "epochs": n_iter,
            "sparcity_loss": lambda_sparsity,
            "noise": rand_noise
        }

    # nets
    net = create_net()
    mask_net = create_mask_net().to(device)

    checkpoint = torch.load(tools.checkpoint_path, map_location=lambda storage, loc: storage)
    mask_net.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    print("start running with {}".format(run_name))

    for i in range(n_iter):
        optimizer.zero_grad()

        batch = []
        mask_batch = []
        for b in range(batch_size):
            idx = random.randint(0, train_size-1)
            augment = augmentations.crop(crop_size)
            batch.append(augment(ys[idx]))

            with torch.no_grad():
                pred = mask_net.netG(ys[idx])
            # the output of the network is logits (i.e., no activaiton). So we apply sigmoid and get probabilites
            pred_p = torch.sigmoid(pred)
            mask = augment((pred_p > 0.5).float())
            mask_batch.append(mask)

        mask_batch = torch.cat(mask_batch)
        iter_input = torch.cat(batch)

        y_ref = iter_input
        iter_input = iter_input #+ torch.randn(iter_input.shape).to(device)*random.uniform(0, 0.1)

        x = net(iter_input)
        y_recon = F.conv2d(x, w)

        loss_sparsity = torch.mean(torch.abs(x))
        loss_recon = F.mse_loss(y_recon, y_ref)
        loss_mask = calculate_mask_loss(x, mask_batch)
        loss = loss_recon + lambda_sparsity * loss_sparsity + lambda_mask * loss_mask
        loss.backward()
        optimizer.step()

        wandb.log({"all iters loss": loss})
        if log and idx == 0:
            wandb.log({"loss": loss})
            wandb.log({"loss reconstructin": loss_recon})
            wandb.log({"loss sparcity": loss_sparsity})
            wandb.log({"loss mask": loss_mask})

        if i % 100 == 0:
            print(f'Iteration {i}: loss={loss.item():.4f} | '
                  f'sparsity={loss_sparsity.item():.4f} | recon={loss_recon.item():.4f} | mask={loss_mask.item():.4f}')

    full_x = net(ys[0])

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
    ys = [tools.load_y(tools.OTHERS_PATH, ['{} {}'.format(ch, str(i)) for ch in tools.MULTI]) for i in range(9)] ##TODO: change this back to 9
    other_ys = [tools.load_y(tools.OTHERS_PATH, ['{} {}'.format(ch, str(i)) for ch in tools.MULTI]) for i in range(1, 9)]
    gt = tools.load_y(tools.PROJ_PATH, tools.CHANNELS, stack=False)
    w = tools.load_w()
    opt(w, y, gt, other_ys, ys, lambda_sparsity=args.lambda_sparsity, lambda_mask=args.lambda_mask,
        channels_names=channels_names,
        n_iter=args.n_iter, lr=args.lr, crop_size=args.crop_size, rand_noise=args.noise, log=args.log,
        batch_size=args.batch_size, train_size=args.train_size, save_path=tools.save_path.split('compressed-sensing-rotation/')[-1])


if __name__ == '__main__':

    args = parser.parse_args()

    run(args)
