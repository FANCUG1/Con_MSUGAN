import os
import torch
from config import get_arguments
import functions


def generate_samples(netG, reals_shapes, noise_amp, n=10):
    dir2save_parent = os.path.join(dir2save, 'random_samples')

    try:
        os.makedirs(dir2save_parent)
    except OSError:
        pass

    for idx in range(n):
        noise = functions.sample_random_noise(opt.train_stages - 1, reals_shapes, opt)
        sample = netG(noise, reals_shapes, noise_amp)
        functions.save_generate_image_as_txt(sample, idx, dir2save_parent)


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='with GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='generated random samples', default=10)
    parser.add_argument('--naive_img', help='naive input image (harmonization or editing)', default='')

    opt = parser.parse_args(['--model_dir', './TrainedModels/TI_3D_delta/test_for_concurrent_multistage_UNet_GAN'])
    _gpu = opt.gpu
    _naive_img = opt.naive_img
    __model_dir = opt.model_dir

    opt = functions.load_config(opt=opt)

    opt.gpu = _gpu
    opt.naive_img = _naive_img
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    netG = torch.load('%s/G.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
    fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
    reals = torch.load('%s/reals.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir, map_location='cuda:{}'.format(torch.cuda.current_device()))

    reals_shapes = [r.shape for r in reals]

    with torch.no_grad():
        generate_samples(netG, reals_shapes, noise_amp, n=20)

