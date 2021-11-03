"""Functions for training a SinGAN."""

import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import math
import numpy as np
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize


def train(opt, Gs, word_bank, converter, trba_net, resnet, emb_fixed, height, width, reals, NoiseAmp):
  real_ = functions.read_image(opt)
  in_s = 0
  scale_num = 0
  real = imresize(real_, opt.scale1, opt)
  reals = functions.creat_reals_pyramid(real, reals, opt)
  nfc_prev = 0

  # Generate text image embedding.
  fake_text = np.random.choice(word_bank, 1)[0].lower()
  print('Fake text:', fake_text)
  emb_t = functions.generate_text_emb(fake_text, width, height, resnet)

  while scale_num < opt.stop_scale + 1:
    opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
    opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

    opt.out_ = functions.generate_dir2save(opt)
    opt.outf = '%s/%d' % (opt.out_, scale_num)
    try:
      os.makedirs(opt.outf)
    except OSError:
      pass

    plt.imsave(
      '%s/real_scale.png' %  (opt.outf), 
      functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

    D_curr, G_curr = init_models(opt)
    if (nfc_prev == opt.nfc):
      G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
      D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))

    in_s, G_curr = train_single_scale(
      D_curr, G_curr, converter, trba_net, fake_text, emb_t, emb_fixed, reals, Gs, in_s, NoiseAmp, opt)

    G_curr = functions.reset_grads(G_curr, False)
    G_curr.eval()
    D_curr = functions.reset_grads(D_curr, False)
    D_curr.eval()

    Gs.append(G_curr)
    NoiseAmp.append(opt.noise_amp)

    torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    torch.save(reals, '%s/reals.pth' % (opt.out_))
    torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

    scale_num += 1
    nfc_prev = opt.nfc
    del D_curr, G_curr


def train_single_scale(netD, netG, converter, trba_net, fake_text, emb_t, emb_fixed, reals, Gs, in_s, NoiseAmp, opt, centers=None):
  scale = len(Gs)
  real = reals[len(Gs)]
  opt.nzx = real.shape[-1]
  opt.nzy = real.shape[-2]
  emb_t = F.interpolate(emb_t, (opt.nzy, opt.nzx))
  emb_fixed = F.interpolate(emb_fixed, (opt.nzy, opt.nzx))
  opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
  pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
  pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
  m_noise = nn.ZeroPad2d(int(pad_noise))
  m_image = nn.ZeroPad2d(int(pad_image))
  emb_t = m_noise(emb_t)
  emb_fixed = m_noise(emb_fixed)

  alpha = opt.alpha

  # Setup optimizer.
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
  schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
  schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

  errD2plot = []
  errG2plot = []
  D_real2plot = []
  D_fake2plot = []
  z_opt2plot = []
  err_ocrplot = []

  for epoch in range(opt.niter):
    ############################
    # (1) Update D network: maximize D(x) + D(G(z))
    ###########################
    for j in range(opt.Dsteps):
      # Train with real example.
      netD.zero_grad()

      output = netD(real).to(opt.device)
      errD_real = -output.mean()
      errD_real.backward(retain_graph=True)
      D_x = -errD_real.item()

      # Train with fake example.
      if (j == 0) & (epoch == 0):
        if (Gs == []):
          prev = torch.full([1, opt.nc_im, opt.nzy, opt.nzx], 0, device=opt.device)
          in_s = prev
          prev = m_image(prev)
          z_prev = torch.full([1, opt.nc_im, opt.nzy, opt.nzx], 0, device=opt.device)
          z_prev = m_noise(z_prev)
          opt.noise_amp = 1
        else:
          prev = draw_concat(Gs, emb_t, reals, NoiseAmp, in_s, 'rand', m_noise, m_image, opt)
          prev = m_image(prev)
          z_prev = draw_concat(Gs, emb_fixed, reals, NoiseAmp, in_s, 'rec', m_noise, m_image, opt)
          criterion = nn.MSELoss()
          RMSE = torch.sqrt(criterion(real, z_prev))
          opt.noise_amp = opt.noise_amp_init * RMSE
          z_prev = m_image(z_prev)
      else:
        prev = draw_concat(Gs, emb_t, reals, NoiseAmp, in_s, 'rand', m_noise, m_image,opt)
        prev = m_image(prev)

      z_in = torch.cat([emb_t, torch.zeros((1, 3, emb_t.shape[2], emb_t.shape[3]))], axis=1)
      fake = netG(z_in, prev)
      gradient_penalty = 0.0
      errD_fake = 0.0
      if scale >= opt.patch_scale:
        output = netD(fake.detach())
        errD_fake = output.mean()
        errD_fake.backward(retain_graph=True)
        D_G_z = output.mean().item()
        D_fake2plot.append(D_G_z)
      gradient_penalty = functions.calc_gradient_penalty(
        netD, real, fake, opt.lambda_grad, opt.device)
      gradient_penalty.backward()
      err_ocr = functions.calc_err_ocr(fake_text, fake, converter, trba_net, opt)
      err_ocr.backward()
      errD = err_ocr + errD_real + errD_fake + gradient_penalty
      optimizerD.step()

    err_ocrplot.append(err_ocr.detach())
    errD2plot.append(errD.detach())

    ############################
    # (2) Update G network: maximize D(G(z))
    ###########################

    for j in range(opt.Gsteps):
      netG.zero_grad()
      output = netD(fake)
      errG = -output.mean()
      errG.backward(retain_graph=True)
      if alpha != 0:
        loss = nn.MSELoss()
        z_in_fixed = torch.cat([emb_fixed, torch.zeros((1, 3, emb_fixed.shape[2], emb_fixed.shape[3]))], axis=1)
        rec_loss = alpha * loss(netG(z_in_fixed, z_prev), real)
        rec_loss.backward(retain_graph=True)
        rec_loss = rec_loss.detach()
      else:
        rec_loss = 0

      optimizerG.step()

    errG2plot.append(errG.detach() + rec_loss)
    D_real2plot.append(D_x)
    z_opt2plot.append(rec_loss)

    if epoch % 25 == 0 or epoch == (opt.niter - 1):
      print('scale %d: [%d/%d]' % (len(Gs), epoch, opt.niter))

    if epoch % 500 == 0 or epoch == (opt.niter - 1):
      plt.imsave(
        '%s/fake_sample.png' % (opt.outf), 
        functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
      plt.imsave(
        '%s/G(z_opt).png' % (opt.outf), 
        functions.convert_image_np(netG(z_in.detach(), z_prev).detach()), vmin=0, vmax=1)
      torch.save(emb_fixed, '%s/emb_fixed.pth' % (opt.outf))

      functions.save_losscurve(errD2plot, '%s/errD2plot.png' % (opt.outf))
      functions.save_losscurve(errG2plot, '%s/errG2plot.png' % (opt.outf))
      functions.save_losscurve(D_real2plot, '%s/D_real2plot.png' % (opt.outf))
      functions.save_losscurve(D_fake2plot, '%s/D_fake2plot.png' % (opt.outf))
      functions.save_losscurve(z_opt2plot, '%s/z_opt2plot.png' % (opt.outf))
      functions.save_losscurve(err_ocrplot, '%s/err_ocrplot.png' % (opt.outf))

    schedulerD.step()
    schedulerG.step()

  functions.save_networks(netG, netD, emb_fixed, opt)
  return in_s, netG    


def draw_concat(Gs, emb_t, reals, NoiseAmp, in_s, mode, m_noise, m_image, opt):
  G_z = in_s
  if len(Gs) > 0:
    if mode == 'rand':
      count = 0
      pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
      for (G, real_curr, real_next, noise_amp) in zip(Gs, reals, reals[1:], NoiseAmp):
        G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
        G_z = m_image(G_z)
        emb_t = F.interpolate(emb_t, (real_curr.shape[2], real_curr.shape[3]))
        emb_t = m_noise(emb_t)
        z_in = torch.cat([emb_t, G_z], axis=1)
        G_z = G(z_in, G_z)
        G_z = imresize(G_z.detach(), 1 / opt.scale_factor, opt)
        G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
        count += 1
    if mode == 'rec':
      count = 0
      for (G, real_curr, real_next, noise_amp) in zip(Gs, reals, reals[1:], NoiseAmp):
        G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
        G_z = m_image(G_z)
        emb_t = F.interpolate(emb_t, (real_curr.shape[2], real_curr.shape[3]))
        emb_t = m_noise(emb_t)
        z_in = torch.cat([emb_t, G_z], axis=1)
        G_z = G(z_in, G_z)
        G_z = imresize(G_z.detach(), 1 / opt.scale_factor, opt)
        G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
        count += 1
  return G_z


def init_models(opt):
  # Initialize generator.
  netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
  netG.apply(models.weights_init)
  if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
  print(netG)

  # Initialize discriminator.
  netD = models.WDiscriminator(opt).to(opt.device)
  netD.apply(models.weights_init)
  if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
  print(netD)

  return netD, netG
