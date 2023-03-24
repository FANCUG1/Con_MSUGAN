import os
import torch
import torch.nn as nn
import torch.optim as optim

import functions
import U_Net_2D
import matplotlib.pyplot as plt


def init_G(opt):
    netG = U_Net_2D.GrowingGenerator(opt).to(opt.device)
    netG.apply(U_Net_2D.weights_init)
    return netG


def init_D(opt):
    netD = U_Net_2D.Discriminator(dim=32).to(opt.device)
    netD.apply(U_Net_2D.weights_init)
    return netD


def generate_samples(netG, opt, depth, noise_amp, reals):
    opt.out_ = functions.generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    n = 5

    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # 开始生成模型
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(depth, reals_shapes, opt)
            sample = netG.forward(noise, reals_shapes, noise_amp)
            plt.imsave('%s/fake_sample_%d.png' % (dir2save, idx + 1), functions.convert_image_np(sample.detach()), vmin=0, vmax=1)


def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))

    real = functions.read_image(opt)
    real = functions.adjust_scale2image(real, opt)
    reals = functions.create_reals_pyramid(real, opt)
    print("Training on image pyramid: {}".format(r.shape for r in reals))
    print("")

    generator = init_G(opt)

    fixed_noise = []
    noise_amp = []

    for scale_num in range(opt.stop_scale + 1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            print(OSError)
            pass

        plt.imsave('%s/real_scale_%d.png' % (opt.outf, scale_num + 1), functions.convert_image_np(inp=reals[scale_num]), vmin=0, vmax=1)

        d_curr = init_D(opt)

        if scale_num > 0:
            d_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))
            generator.init_next_stage()

        fixed_noise, noise_amp, generator, d_curr = train_single_scale(netD=d_curr, netG=generator, reals=reals,
                                                                       fixed_noise=fixed_noise, noise_amp=noise_amp,
                                                                       opt=opt, depth=scale_num)

        if scale_num == opt.stop_scale:
            torch.save(fixed_noise, '%s/fixed_noise.pth' % opt.out_)
            torch.save(generator, '%s/G.pth' % opt.out_)
            torch.save(reals, '%s/reals.pth' % opt.out_)
            torch.save(noise_amp, '%s/noise_amp.pth' % opt.out_)

        del d_curr


def train_single_scale(netD, netG, reals, fixed_noise, noise_amp, opt, depth):

    reals_shapes = [real.shape for real in reals]

    real = reals[depth].to(opt.device)

    alpha = opt.alpha

    if depth == 0:
        z_opt = reals[0].to(opt.device)

    else:
        z_opt = functions.generate_noise(size=[opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                         device=opt.device)

    fixed_noise.append(z_opt.detach())

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    parameter_list = [
        {"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale ** (len(netG.body[-opt.train_depth:]) - 1 - idx))}
        for idx, block in enumerate(netG.body[-opt.train_depth:], 0)]


    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8 * opt.niter],
                                                      gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8 * opt.niter],
                                                      gamma=opt.gamma)

    if depth == 0:
        noise_amp.append(1)
    else:
        noise_amp.append(0)
        z_reconstruction = netG.forward(fixed_noise, reals_shapes, noise_amp)

        criterion = nn.MSELoss()
        rec_loss = criterion(z_reconstruction, real)

        RMSE = torch.sqrt(rec_loss).detach()
        _noise_amp = opt.noise_amp_init * RMSE
        noise_amp[-1] = _noise_amp

    print('stage: [{}/{}]'.format(depth, opt.stop_scale))
    for iter in range(opt.niter):

        noise = functions.sample_random_noise(depth=depth, reals_shapes=reals_shapes, opt=opt)

        for j in range(opt.Dsteps):
            netD.zero_grad()
            output = netD(real)
            errD_real = -output.mean()

            fake = netG.forward(noise, reals_shapes, noise_amp)
            output = netD(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            errD_total = errD_fake + errD_real + gradient_penalty
            errD_total.backward(retain_graph=True)
            optimizerD.step()

        fake = netG(noise, reals_shapes, noise_amp)
        output = netD(fake)
        errG = -output.mean()

        if alpha != 0:
            loss = nn.MSELoss()
            rec = netG.forward(fixed_noise, reals_shapes, noise_amp)
            rec_loss = alpha * loss(rec, real)
        else:
            rec_loss = 0

        netG.zero_grad()
        errG_total = errG + rec_loss
        errG_total.backward(retain_graph=True)

        optimizerG.step()

        if iter % 100 == 0 or iter + 1 == opt.niter:
            print('errD_real_loss: %.4f' % errD_real.item())
            print('errD_fake_loss: %.4f' % errD_fake.item())
            print('gradient_penalty_loss: %.4f' % gradient_penalty.item())
            print('generator_loss: %.4f' % errG.item())
            print('reconstruction_loss: %.4f' % rec_loss.item())
            print('\n')
        if iter + 1 == opt.niter:
            generate_samples(netG, opt, depth, noise_amp, reals)

        schedulerD.step()
        schedulerG.step()

    functions.save_networks(netG, netD, z_opt, opt)
    return fixed_noise, noise_amp, netG, netD
