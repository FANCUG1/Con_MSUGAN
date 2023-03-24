import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import imresize


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def training(x, block):
    x = block[0](x)
    x = block[1](x)
    x = block[2](x)
    x1 = x

    x = block[3](x)
    x = block[4](x)
    x = block[5](x)
    x2 = x

    x = block[6](x)
    x = block[7](x)
    x = block[8](x)
    x3 = x

    x = block[9](x)
    x = block[10](x)
    x = block[11](x)
    x4 = x

    x = block[12](x)

    x = block[13](x)
    x = block[14](x)
    x = block[15](x)
    x = torch.concat([x, x4], dim=1)

    x = block[16](x)
    x = block[17](x)
    x = block[18](x)
    x = torch.concat([x, x3], dim=1)

    x = block[19](x)
    x = block[20](x)
    x = block[21](x)
    x = torch.concat([x, x2], dim=1)

    x = block[22](x)
    x = block[23](x)
    x = block[24](x)
    x = torch.concat([x, x1], dim=1)

    x = block[25](x)
    x = block[26](x)

    return x


def upsample(x, size):
    x_up = F.upsample(input=x, size=size, mode='bilinear', align_corners=True)
    return x_up


class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()
        self.opt = opt
        N = opt.nfc  # 32

        self.body = nn.ModuleList([])

        _first_stage = nn.Sequential(
            nn.Conv2d(in_channels=opt.nc_im, out_channels=N, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=N, out_channels=2 * N, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(2 * N),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=2 * N, out_channels=4 * N, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(4 * N),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=4 * N, out_channels=8 * N, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(8 * N),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=8 * N, out_channels=16 * N, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),

            nn.ConvTranspose2d(in_channels=16 * N, out_channels=8 * N, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(8 * N),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=16 * N, out_channels=4 * N, kernel_size=(4, 4), stride=(1, 1),
                               padding=(1, 1)),
            nn.BatchNorm2d(4 * N),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=8 * N, out_channels=2 * N, kernel_size=(4, 4), stride=(1, 1),
                               padding=(1, 1)),
            nn.BatchNorm2d(2 * N),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=4 * N, out_channels=N, kernel_size=(4, 4), stride=(1, 1),
                               padding=(1, 1)),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels=2 * N, out_channels=opt.nc_im, kernel_size=(4, 4), stride=(1, 1),
                               padding=(1, 1)),
            nn.Tanh(),
        )

        self.body.append(_first_stage)

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x_prev_out = training(noise[0], self.body[0])

        for idx, block in enumerate(self.body[1:], 1):
            if idx < len(self.body[1:]):

                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])

                x_prev_out_2 = x_prev_out_1 + noise[idx] * noise_amp[idx]

                x_prev_out = training(x_prev_out_2, block)

            else:
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])

                x_prev_out_2 = x_prev_out_1 + noise[idx] * noise_amp[idx]

                x_prev_out = training(x_prev_out_2, block)


        return x_prev_out


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.dim = dim

        self.image_to_features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=dim, out_channels=2 * dim, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=2 * dim, out_channels=4 * dim, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=4 * dim, out_channels=8 * dim, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=8 * dim, out_channels=1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.image_to_features(x)
        return x
