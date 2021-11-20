"""Functions for training a SinGAN."""

import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize
import tracemalloc
import linecache
import imageio
import cv2


def display_top(snapshot, key_type='lineno', limit=3):
  snapshot = snapshot.filter_traces((
    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    tracemalloc.Filter(False, "<unknown>"),
  ))
  top_stats = snapshot.statistics(key_type)

  print("Top %s lines" % limit)
  for index, stat in enumerate(top_stats[:limit], 1):
    frame = stat.traceback[0]
    # replace "/path/to/module/file.py" with "module/file.py"
    filename = os.sep.join(frame.filename.split(os.sep)[-2:])
    print("#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024))
    line = linecache.getline(frame.filename, frame.lineno).strip()
    if line:
      print('    %s' % line)

  other = top_stats[limit:]
  if other:
    size = sum(stat.size for stat in other)
    print("%s other: %.1f KiB" % (len(other), size / 1024))
  total = sum(stat.size for stat in top_stats)
  print("Total allocated size: %.1f KiB" % (total / 1024))


def get_top(snapshot, key_type='lineno', limit=3):
  snapshot = snapshot.filter_traces((
    tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
    tracemalloc.Filter(False, "<unknown>"),
  ))
  top_stats = snapshot.statistics(key_type)

  # print("Top %s lines" % limit)
  for index, stat in enumerate(top_stats[:limit], 1):
    frame = stat.traceback[0]
    # replace "/path/to/module/file.py" with "module/file.py"
    filename = os.sep.join(frame.filename.split(os.sep)[-2:])
    # print("#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024))
    line = linecache.getline(frame.filename, frame.lineno).strip()
    # if line:
      # print('    %s' % line)

  other = top_stats[limit:]
  if other:
    size = sum(stat.size for stat in other)
    # print("%s other: %.1f KiB" % (len(other), size / 1024))
  total = sum(stat.size for stat in top_stats)
  # print("Total allocated size: %.1f KiB" % (total / 1024))
  return total


def train(opt, Gs, word_bank, converter, trba_net, resnet, emb_fixed, height, width, reals):
  real_ = functions.read_image(opt)
  in_s = [0, 0]
  scale_num = 0
  real = imresize(real_, opt.scale1, opt)
  reals = functions.creat_reals_pyramid(real, reals, opt)
  nfc_prev = 0

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
      D_curr, G_curr, resnet, converter, trba_net, word_bank, emb_fixed, height, width, reals, Gs, in_s, opt)

    G_curr = functions.reset_grads(G_curr, False)
    G_curr.eval()
    D_curr = functions.reset_grads(D_curr, False)
    D_curr.eval()

    Gs.append(G_curr)

    torch.save(Gs, '%s/Gs.pth' % (opt.out_))
    torch.save(reals, '%s/reals.pth' % (opt.out_))

    scale_num += 1
    nfc_prev = opt.nfc
    del D_curr, G_curr


def train_single_scale(netD, netG, resnet, converter, trba_net, word_bank, emb_fixed, height, width, reals, Gs, in_s, opt, centers=None):
  total_mem = []
  total_time = []
  recons = []
  fakes = []
  scale = len(Gs)
  real = reals[len(Gs)]
  opt.nzx = real.shape[-1]
  opt.nzy = real.shape[-2]
  opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride
  pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
  pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
  m_noise = nn.ZeroPad2d(int(pad_noise))
  m_image = nn.ZeroPad2d(int(pad_image))
  emb_fixed = F.interpolate(emb_fixed, (opt.nzy, opt.nzx))

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

  t0 = time.perf_counter()
  for (epoch, (fake_text_img, fake_text)) in zip(range(opt.niter), word_bank):
    t1 = time.perf_counter()
    tracemalloc.start()

    # Generate text image embedding.
    fake_text = fake_text[0]
    print('Fake text:', fake_text, flush=True)
    if opt.use_resnet:
      emb_t = functions.generate_text_emb(fake_text_img, resnet)
      emb_t = F.interpolate(emb_t, (opt.nzy, opt.nzx))
    else:
      emb_t = F.interpolate(fake_text_img.float(), (opt.nzy, opt.nzx))
    # emb_t = m_noise(emb_t)
    
    ############################
    # (1) Update D network: maximize D(x) + D(G(z))
    ############################
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
          prev = emb_t
          z_prev = emb_fixed
          in_s = [prev, z_prev]
          prev = m_image(prev)
          z_prev = m_noise(z_prev)
        else:
          prev = draw_concat(Gs, emb_t, reals, in_s[0], 'rand', m_noise, m_image, opt)
          prev = m_image(prev)
          z_prev = draw_concat(Gs, emb_fixed, reals, in_s[1], 'rec', m_noise, m_image, opt)
          z_prev = m_image(z_prev)
      else:
        prev = draw_concat(Gs, emb_t, reals, in_s[0], 'rand', m_noise, m_image, opt)
        prev = m_image(prev)

      z_in = emb_t
      z_in = z_in.to(opt.device)
      prev = prev.to(opt.device)
      fake = netG(prev)
      gradient_penalty = 0.0
      errD_fake = 0.0
      if scale >= opt.patch_scale:
        output = netD(fake)
        errD_fake = output.mean()
        errD_fake.backward(retain_graph=True)
        D_G_z = output.mean().item()
        D_fake2plot.append(D_G_z)
      gradient_penalty = functions.calc_gradient_penalty(
        netD, real, fake, opt.lambda_grad, opt.device)
      gradient_penalty.backward()
      err_ocr = functions.calc_err_ocr(fake_text, fake, converter, trba_net, opt)
      err_ocr.backward()
      fake = fake.detach()
      errD = err_ocr + errD_real + errD_fake + gradient_penalty
      optimizerD.step()

    err_ocrplot.append(err_ocr.detach())
    errD2plot.append(errD.detach())

    ############################
    # (2) Update G network: maximize D(G(z))
    ############################

    for j in range(opt.Gsteps):
      netG.zero_grad()
      output = netD(fake)
      errG = -output.mean()
      errG.backward(retain_graph=True)
      if alpha != 0:
        loss = nn.MSELoss()
        z_in_fixed = emb_fixed
        z_in_fixed = z_in_fixed.to(opt.device)
        z_prev = z_prev.to(opt.device)
        recon = netG(z_prev)
        rec_loss = alpha * loss(recon, real)
        rec_loss.backward(retain_graph=True)
        rec_loss = rec_loss.detach()
        err_ocr = functions.calc_err_ocr(opt.text, recon, converter, trba_net, opt)
        err_ocr.backward()
      else:
        rec_loss = 0

      optimizerG.step()

    errG2plot.append(errG.detach() + rec_loss + err_ocr)
    D_real2plot.append(D_x)
    z_opt2plot.append(rec_loss)

    if epoch % 25 == 0 or epoch == (opt.niter - 1):
      t2 = time.perf_counter()
      total_time.append(t1 - t0)
      print('scale %d: [%d/%d] - %.2fs / %.2fs' % (len(Gs), epoch, opt.niter, t2 - t1, t2 - t0), flush=True)

    if epoch % opt.save_freq == 0 or epoch == (opt.niter - 1):
      fake_img = functions.convert_image_np(fake.detach())
      plt.imsave(
        '%s/fake_sample.png' % (opt.outf), fake_img, vmin=0, vmax=1)
      fakes.append(fake_img)
      recon_img = functions.convert_image_np(netG(z_prev).detach())
      plt.imsave(
        '%s/G(z_opt).png' % (opt.outf), recon_img, vmin=0, vmax=1)
      recons.append(recon_img)
      torch.save(emb_fixed, '%s/emb_fixed.pth' % (opt.outf))

      functions.save_losscurve(errD2plot, '%s/errD2plot.png' % (opt.outf))
      functions.save_losscurve(errG2plot, '%s/errG2plot.png' % (opt.outf))
      functions.save_losscurve(D_real2plot, '%s/D_real2plot.png' % (opt.outf))
      functions.save_losscurve(D_fake2plot, '%s/D_fake2plot.png' % (opt.outf))
      functions.save_losscurve(z_opt2plot, '%s/z_opt2plot.png' % (opt.outf))
      functions.save_losscurve(err_ocrplot, '%s/err_ocrplot.png' % (opt.outf))
      functions.save_losscurve(total_mem, '%s/mem_usage.png' % (opt.outf))
      functions.save_losscurve(total_time, '%s/time_stats.png' % (opt.outf))

    schedulerD.step()
    schedulerG.step()
    snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
    top_mem = get_top(snapshot)
    total_mem.append(top_mem)

  functions.save_networks(netG, netD, emb_fixed, opt)
  imageio.mimsave('%s/fake.gif' % (opt.outf), fakes)
  imageio.mimsave('%s/recon.gif' % (opt.outf), recons)
  return in_s, netG    


def draw_concat(Gs, emb_t, reals, in_s, mode, m_noise, m_image, opt):
  G_z = in_s
  if len(Gs) > 0:
    if mode == 'rand':
      count = 0
      pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
      for (G, real_curr, real_next) in zip(Gs, reals, reals[1:]):
        G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
        G_z = m_image(G_z).to(opt.device)
        z_in = G_z
        G_z = G(G_z)
        G_z = imresize(G_z.detach(), 1 / opt.scale_factor, opt)
        G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
        count += 1
    if mode == 'rec':
      count = 0
      for (G, real_curr, real_next) in zip(Gs, reals, reals[1:]):
        G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
        G_z = m_image(G_z).to(opt.device)
        z_in = G_z
        G_z = G(G_z)
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
  print(netG, flush=True)

  # Initialize discriminator.
  netD = models.WDiscriminator(opt).to(opt.device)
  netD.apply(models.weights_init)
  if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
  print(netD, flush=True)

  return netD, netG
