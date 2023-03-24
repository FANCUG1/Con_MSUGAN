import torch
import torch.nn as nn
import math
import os
import numpy as np
from imresize import imresize


def post_config(opt):
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.noise_amp_init = opt.noise_amp

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def generate_dir2save(opt):
    training_image_name = opt.input_name[:-4].split('/')[-1]
    dir2save = 'TrainedModels/{}/'.format(training_image_name)
    dir2save += 'test_for_concurrent_multistage_UNet_GAN'
    return dir2save


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def move_to_gpu(t):
    if torch.cuda.is_available():
        t = t.to(torch.device('cuda'))
    return t


def np2torch(x, opt):
    x = x[:, :, :, None]
    x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not (opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not (opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def tackle_data(curr_real):
    x = curr_real.size()[2]
    y = curr_real.size()[3]
    curr_real = curr_real.detach().cpu().numpy()

    curr_real = curr_real.reshape((x, y))
    curr_real = curr_real.reshape((x * y, 1))

    for i in range(len(curr_real)):
        if curr_real[i] <= 0:
            curr_real[i] = -1
        else:
            curr_real[i] = 1

    curr_real = curr_real.reshape((x, y))
    curr_real = curr_real.reshape((1, 1, x, y))
    curr_real = torch.from_numpy(curr_real)
    return curr_real


def read_image(opt):
    f = open('%s' % opt.input_name)
    data = f.readlines()
    data_size = data[0].replace('\n', '').split(' ')
    data_size = list(map(eval, data_size))
    x = []
    for i in range(3, len(data)):
        x.append(int(data[i]))
    x = np.array(x)
    x = x.reshape((data_size[0], data_size[1], 1))
    x = np2torch(x, opt)
    return x


def adjust_scale2image(real_, opt):
    opt.scale1 = []
    opt.scale1.append(min(opt.max_size / real_.shape[2], 1))
    opt.scale1.append(min(opt.max_size / real_.shape[3], 1))

    real = imresize(real_, opt.scale1, opt)

    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = []
    opt.scale_factor.append(math.pow(opt.min_size / real.shape[2], 1 / opt.stop_scale))
    opt.scale_factor.append(math.pow(opt.min_size / real.shape[3], 1 / opt.stop_scale))
    return real


def create_reals_pyramid(real, opt):
    reals = []
    for i in range(opt.stop_scale):
        scale = []
        scale.append(math.pow(opt.scale_factor[0],
                              ((opt.stop_scale - 1) / math.log(opt.stop_scale)) * math.log(opt.stop_scale - i) + 1))
        scale.append(math.pow(opt.scale_factor[1],
                              ((opt.stop_scale - 1) / math.log(opt.stop_scale)) * math.log(opt.stop_scale - i) + 1))
        curr_real = imresize(real, scale, opt)

        curr_real = tackle_data(curr_real)

        reals.append(curr_real)
    reals.append(real)
    return reals


def save_image_as_txt(image, opt, scale_num):
    image_size = list(image.size())
    image = image.detach().cpu().numpy()
    x = image_size[2]
    y = image_size[3]

    image = image.reshape((x, y))
    image = image.reshape((x * y, 1))

    file = open('%s/real_image_with_different_scale_%d.txt' % (opt.outf, scale_num), 'w')
    file.write(str(x) + ' ' + str(y) + ' ' + str(1) + '\n')
    file.write(str(1) + '\n')
    file.write('facies' + '\n')

    for i in range(len(image)):
        temp = str(float(image[i]))
        file.write(temp + '\n')

    file.close()


def upsampling(im, sx, sy):
    m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
    return m(im)


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    elif type == 'gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1 + noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    else:
        raise NotImplementedError
    return noise


def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        noise.append(
            generate_noise(size=[opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], device=opt.device).detach())
    return noise


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)

    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates = interpolates.to(device)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def save_generate_image_as_txt(image, iter, path):
    image_size = list(image.size())
    image = image.detach().cpu().numpy()
    x = image_size[2]
    y = image_size[3]

    image = image.reshape((x, y))
    image = image.reshape((x * y, 1))

    file = open('%s/generated_image_%d.txt' % (path, iter), 'w')
    file.write(str(x) + ' ' + str(y) + ' ' + str(1) + '\n')
    file.write(str(1) + '\n')
    file.write('facies' + '\n')

    for i in range(len(image)):
        temp = str(float(image[i]))
        file.write(temp + '\n')

    file.close()


def save_networks(netG, netDs, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))


def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt
