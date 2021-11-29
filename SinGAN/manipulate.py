"""Functions for manipulating images."""

from __future__ import print_function

import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
import SinGAN.models as models
from config import get_arguments

def generate_gif(Gs, Zs, reals, NoiseAmp, opt, alpha=0.1, beta=0.9, start_scale=2, fps=10):
  in_s = torch.full(Zs[0].shape, 0, device=opt.device)
  images_cur = []
  count = 0

  for (G, Z_opt, noise_amp, real) in zip(Gs, Zs, NoiseAmp, reals):
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    nzx = Z_opt.shape[2]
    nzy = Z_opt.shape[3]
    m_image = nn.ZeroPad2d(int(pad_image))
    images_prev = images_cur
    images_cur = []
    if count == 0:
      z_rand = functions.generate_noise([1, nzx, nzy], device=opt.device)
      z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
      z_prev1 = 0.95 * Z_opt + 0.05 * z_rand
      z_prev2 = Z_opt
    else:
      z_prev1 = 0.95 * Z_opt + 0.05 * functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
      z_prev2 = Z_opt

    for i in range(0, 100, 1):
      if count == 0:
        z_rand = functions.generate_noise([1,nzx,nzy], device=opt.device)
        z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
        diff_curr = beta * (z_prev1 - z_prev2)+(1 - beta) * z_rand
      else:
        diff_curr = (
          beta * (z_prev1 - z_prev2) + (1 - beta) * 
          (functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device))
          )

      z_curr = alpha * Z_opt + (1 - alpha) * (z_prev1 + diff_curr)
      z_prev2 = z_prev1
      z_prev1 = z_curr

      if images_prev == []:
        I_prev = in_s
      else:
        I_prev = images_prev[i]
        I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
        I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
        I_prev = m_image(I_prev)
      if count < start_scale:
        z_curr = Z_opt

      z_in = noise_amp * z_curr + I_prev
      I_curr = G(z_in.detach(), I_prev)

      if (count == len(Gs) - 1):
        I_curr = functions.denorm(I_curr).detach()
        I_curr = I_curr[0, :, :, :].cpu().numpy()
        I_curr = I_curr.transpose(1, 2, 0) * 255
        I_curr = I_curr.astype(np.uint8)

      images_cur.append(I_curr)
    count += 1
  dir2save = functions.generate_dir2save(opt)
  try:
    os.makedirs('%s/start_scale=%d' % (dir2save,start_scale) )
  except OSError:
    pass
  imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (
    dir2save, start_scale, alpha, beta), images_cur, fps=fps)
  del images_cur


def SinGAN_generate(text, width, height, Gs, reals, opt, n=0):
  n = 0
  fake_text_img = functions.render_text(text, width, height, opt.pad)[None, ...]
  fake_text_img = torch.Tensor(np.transpose(fake_text_img, [0, 3, 1, 2]))

  images = []
  for (G, real) in zip(Gs, reals):
    opt.nzx = real.shape[-1]
    opt.nzy = real.shape[-2]
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_image = nn.ZeroPad2d(int(pad_image))

    # Generate text image embedding.
    if opt.use_resnet:
      emb_t = functions.generate_text_emb(fake_text_img, resnet)
      emb_t = F.interpolate(emb_t, (opt.nzy, opt.nzx))
    else:
      emb_t = F.interpolate(fake_text_img.float(), (opt.nzy, opt.nzx))
    emb_t = emb_t.to(opt.device)
    
    if n == 0:
      prev = emb_t 
    else:
      prev = images[n - 1]
      prev = imresize(prev, 1 / opt.scale_factor, opt)
      prev = prev[:, :, 0:real.shape[2], 0:real.shape[3]]
      prev = F.interpolate(prev, (real.shape[2], real.shape[3]))
    prev = m_image(prev)

    if n <= opt.concat_input:
      z_in = torch.cat([prev, m_image(emb_t)], axis=1)
    else:
      z_in = prev
    z_in = z_in.to(opt.device)
    image = G(z_in.detach())
    images.append(image)
    # if n == len(reals) - 1:
    dir2save = functions.generate_dir2save(opt)
    subfolder = 'lambda_grad=%f,lambda_ocr=%f,niter=%d,pscale=%d,s=%s,ci=%d' % (
      opt.lambda_grad, opt.lambda_ocr, opt.niter, opt.patch_scale, opt.sensitive, opt.concat_input)
    try:
      os.makedirs(os.path.join(dir2save, subfolder))
    except OSError:
      pass
    plt.imsave(
      '%s/%s/%s_%d.png' % (dir2save, subfolder, text, n), 
      functions.convert_image_np(image.detach()), vmin=0, vmax=1)
    n += 1

  return image