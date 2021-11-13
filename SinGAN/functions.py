"""Training utils."""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import math
from skimage import io as img
from skimage import color, morphology, filters
from SinGAN.imresize import imresize
import os
import random
import cv2
from sklearn.cluster import KMeans
from PIL import Image, ImageOps, ImageFont, ImageDraw
from trba_utils import AlignCollate


def read_image(opt):
  x = img.imread('%s%s' % (opt.input_img,opt.ref_image))
  return np2torch(x)


def denorm(x):
  out = (x + 1) / 2
  return out.clamp(0, 1)


def norm(x):
  out = (x - 0.5) * 2
  return out.clamp(-1, 1)


def convert_image_np(inp):
  if inp.shape[1] == 3:
    inp = denorm(inp)
    inp = move_to_cpu(inp[-1, :, :, :])
    inp = inp.numpy().transpose((1, 2, 0))
  else:
    inp = denorm(inp)
    inp = move_to_cpu(inp[-1, -1, :, :])
    inp = inp.numpy().transpose((0, 1))

  inp = np.clip(inp, 0, 1)
  return inp


def save_image(real_cpu, receptive_field, ncs, epoch_num, file_name):
  fig, ax = plt.subplots(1)
  if ncs == 1:
    ax.imshow(real_cpu.view(real_cpu.size(2), real_cpu.size(3)), cmap='gray')
  else:
    ax.imshow(convert_image_np(real_cpu.cpu()))
  rect = patches.Rectangle(
    (0, 0), receptive_field, receptive_field, 
    linewidth=5, edgecolor='r', facecolor='none')
  ax.add_patch(rect)
  ax.axis('off')
  plt.savefig(file_name)
  plt.close(fig)


def convert_image_np_2d(inp):
  inp = denorm(inp)
  inp = inp.numpy()
  return inp


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
  if type == 'gaussian':
    noise = torch.randn(
      num_samp, size[0], round(size[1] / scale), round(size[2] / scale), device=device)
    noise = upsampling(noise, size[1], size[2])
  if type =='gaussian_mixture':
    noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device) + 5
    noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    noise = noise1 + noise2
  if type == 'uniform':
    noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
  return noise


def plot_learning_curves(G_loss, D_loss, epochs, label1, label2, name):
  fig, ax = plt.subplots(1)
  n = np.arange(0, epochs)
  plt.plot(n, G_loss, n, D_loss)
  plt.xlabel('Epochs')
  plt.legend([label1, label2], loc='upper right')
  plt.savefig('%s.png' % name)
  plt.close(fig)


def plot_learning_curve(loss, epochs, name):
  fig, ax = plt.subplots(1)
  n = np.arange(0, epochs)
  plt.plot(n, loss)
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.savefig('%s.png' % name)
  plt.close(fig)


def upsampling(im, sx, sy):
  m = nn.Upsample(size=[round(sx), round(sy)], mode='bilinear', align_corners=True)
  return m(im)


def reset_grads(model, require_grad):
  for p in model.parameters():
    p.requires_grad_(require_grad)
  return model


def move_to_gpu(t):
  if (torch.cuda.is_available()):
      t = t.to(torch.device('cuda'))
  return t


def move_to_cpu(t):
  t = t.to(torch.device('cpu'))
  return t


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
  alpha = torch.rand(1, 1)
  alpha = alpha.expand(real_data.size())
  alpha = alpha.to(device)

  interpolates = alpha * real_data + ((1 - alpha) * fake_data)
  interpolates = interpolates.to(device)
  interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
  disc_interpolates = netD(interpolates)

  gradients = torch.autograd.grad(
    outputs=disc_interpolates, 
    inputs=interpolates,
    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
    create_graph=True, 
    retain_graph=True, 
    only_inputs=True)[0]

  gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * LAMBDA
  return gradient_penalty


def read_image(opt):
  x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
  x = np2torch(x, opt)
  x = x[:, 0:3, :, :]
  return x


def read_image_dir(dir, opt):
  x = img.imread('%s' % (dir))
  x = np2torch(x, opt)
  x = x[:, 0:3, :, :]
  return x


def np2torch(x, opt):
  if opt.nc_im == 3:
    x = x[:, :, :, None]
    x = x.transpose((3, 2, 0, 1)) / 255
  else:
    x = x.transpose(2, 0, 1)
    x = x[None, :, :, :]
  x = torch.from_numpy(x)
  if not(opt.not_cuda):
    x = move_to_gpu(x)
  x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
  x = norm(x)
  return x


def torch2uint8(x):
  x = x[0, :, :, :]
  x = x.permute((1, 2, 0))
  x = 255 * denorm(x)
  x = x.cpu().numpy()
  x = x.astype(np.uint8)
  return x


def read_image2np(opt):
  x = img.imread('%s/%s' % (opt.input_dir, opt.input_name))
  x = x[:, :, 0:3]
  return x


def save_networks(netG, netD, z, opt):
  torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
  torch.save(netD.state_dict(), '%s/netD.pth' % (opt.outf))
  torch.save(z, '%s/z_opt.pth' % (opt.outf))


def adjust_scales2image(real_, opt):
  opt.num_scales = math.ceil(
    (math.log(
      math.pow(opt.min_size / (min(real_.shape[2], real_.shape[3])), 1), 
      opt.scale_factor_init))) + 1
  scale2stop = math.ceil(
    math.log(
      min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]), 
    opt.scale_factor_init))
  opt.stop_scale = opt.num_scales - scale2stop
  opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
  real = imresize(real_, opt.scale1, opt)
  opt.scale_factor = math.pow(
    opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale))
  scale2stop = math.ceil(
    math.log(
      min([opt.max_size, max([real_.shape[2], real_.shape[3]])]) / max([real_.shape[2], real_.shape[3]]),
      opt.scale_factor_init))
  opt.stop_scale = opt.num_scales - scale2stop
  return real


def adjust_scales2image_SR(real_,opt):
  opt.min_size = 18
  opt.num_scales = int((math.log(
    opt.min_size / min(real_.shape[2], real_.shape[3]), opt.scale_factor_init))) + 1
  scale2stop = int(math.log(
    min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), 
    opt.scale_factor_init))
  opt.stop_scale = opt.num_scales - scale2stop
  opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
  real = imresize(real_, opt.scale1, opt)
  opt.scale_factor = math.pow(
    opt.min_size / (min(real.shape[2], real.shape[3])), 1 / (opt.stop_scale))
  scale2stop = int(math.log(
    min(opt.max_size, max(real_.shape[2], real_.shape[3])) / max(real_.shape[0], real_.shape[3]), 
    opt.scale_factor_init))
  opt.stop_scale = opt.num_scales - scale2stop
  return real


def creat_reals_pyramid(real, reals, opt):
  real = real[:, 0:3, :, :]
  for i in range(0, opt.stop_scale + 1, 1):
    scale = math.pow(opt.scale_factor, opt.stop_scale - i)
    curr_real = imresize(real, scale, opt)
    reals.append(curr_real)
  return reals


def load_trained_pyramid(opt, mode_='train'):
  mode = opt.mode
  opt.mode = 'train'
  if (mode == 'animation_train') | (mode == 'SR_train') | (mode == 'paint_train'):
    opt.mode = mode
  dir = generate_dir2save(opt)
  if(os.path.exists(dir)):
    Gs = torch.load('%s/Gs.pth' % dir)
    Zs = torch.load('%s/Zs.pth' % dir)
    reals = torch.load('%s/reals.pth' % dir)
    NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir)
  else:
    print('No appropriate trained model is exist, please train first.')
  opt.mode = mode
  return Gs, Zs, reals, NoiseAmp


def generate_in2coarsest(reals, scale_v, scale_h, opt):
  real = reals[opt.gen_start_scale]
  real_down = upsampling(real, scale_v * real.shape[2], scale_h * real.shape[3])
  if opt.gen_start_scale == 0:
    in_s = torch.full(real_down.shape, 0, device=opt.device)
  else:  # If n != 0
    in_s = upsampling(real_down, real_down.shape[2], real_down.shape[3])
  return in_s


def generate_dir2save(opt):
  dir2save = None
  if (opt.mode == 'train') | (opt.mode == 'SR_train'):
    dir2save = 'TrainedModels/%s,lambda_grad=%f,lambda_ocr=%f,niter=%d/scale_factor=%f,alpha=%d' % (
      opt.input_name[:-4], opt.lambda_grad, opt.lambda_ocr, opt.niter, opt.scale_factor_init, opt.alpha)
  elif (opt.mode == 'animation_train') :
    dir2save = 'TrainedModels/%s/scale_factor=%f_noise_padding' % (
      opt.input_name[:-4], opt.scale_factor_init)
  elif (opt.mode == 'paint_train') :
    dir2save = 'TrainedModels/%s/scale_factor=%f_paint/start_scale=%d' % (
      opt.input_name[:-4], opt.scale_factor_init, opt.paint_start_scale)
  elif opt.mode == 'random_samples':
    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (
      opt.out,opt.input_name[:-4], opt.gen_start_scale)
  elif opt.mode == 'random_samples_arbitrary_sizes':
    dir2save = '%s/RandomSamples_ArbitrerySizes/%s/scale_v=%f_scale_h=%f' % (
      opt.out,opt.input_name[:-4], opt.scale_v, opt.scale_h)
  elif opt.mode == 'animation':
    dir2save = '%s/Animation/%s' % (opt.out, opt.input_name[:-4])
  elif opt.mode == 'SR':
    dir2save = '%s/SR/%s' % (opt.out, opt.sr_factor)
  elif opt.mode == 'harmonization':
    dir2save = '%s/Harmonization/%s/%s_out' % (
      opt.out, opt.input_name[:-4], opt.ref_name[:-4])
  elif opt.mode == 'editing':
    dir2save = '%s/Editing/%s/%s_out' % (
      opt.out, opt.input_name[:-4], opt.ref_name[:-4])
  elif opt.mode == 'paint2image':
    dir2save = '%s/Paint2image/%s/%s_out' % (
      opt.out, opt.input_name[:-4], opt.ref_name[:-4])
    if opt.quantization_flag:
      dir2save = '%s_quantized' % dir2save
  return dir2save


def post_config(opt):
  # Initialize fixed parameters.
  opt.device = torch.device('cpu' if opt.not_cuda else 'cuda:0')
  opt.niter_init = opt.niter
  opt.noise_amp_init = opt.noise_amp
  opt.nfc_init = opt.nfc
  opt.min_nfc_init = opt.min_nfc
  opt.scale_factor_init = opt.scale_factor
  opt.out_ = 'TrainedModels/%s/scale_factor=%f/' % (
    opt.input_name[:-4], opt.scale_factor)
  if opt.mode == 'SR':
    opt.alpha = 100

  if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
  print('Random Seed: ', opt.manualSeed)
  random.seed(opt.manualSeed)
  torch.manual_seed(opt.manualSeed)
  if torch.cuda.is_available() and opt.not_cuda:
    print('WARNING: You have a CUDA device, so you should probably run with --cuda.')
  return opt


def calc_init_scale(opt):
  in_scale = math.pow(1 / 2, 1 / 3)
  iter_num = round(math.log(1 / opt.sr_factor, in_scale))
  in_scale = pow(opt.sr_factor, 1 / iter_num)
  return in_scale, iter_num


def quant(prev,device):
  arr = prev.reshape((-1, 3)).cpu()
  kmeans = KMeans(n_clusters=5, random_state=0).fit(arr)
  labels = kmeans.labels_
  centers = kmeans.cluster_centers_
  x = centers[labels]
  x = torch.from_numpy(x)
  x = move_to_gpu(x)
  x = x.type(torch.cuda.FloatTensor) if () else x.type(torch.FloatTensor)
  x = x.view(prev.shape)
  return x,centers


def quant2centers(paint, centers):
  arr = paint.reshape((-1, 3)).cpu()
  kmeans = KMeans(n_clusters=5, init=centers, n_init=1).fit(arr)
  labels = kmeans.labels_
  x = centers[labels]
  x = torch.from_numpy(x)
  x = move_to_gpu(x)
  x = x.type(torch.cuda.FloatTensor) if torch.cuda.is_available() else x.type(torch.FloatTensor)
  x = x.view(paint.shape)
  return x


def dilate_mask(mask, opt):
  if opt.mode == "harmonization":
    element = morphology.disk(radius=7)
  if opt.mode == "editing":
    element = morphology.disk(radius=20)
  mask = torch2uint8(mask)
  mask = mask[:, :, 0]
  mask = morphology.binary_dilation(mask, selem=element)
  mask = filters.gaussian(mask, sigma=5)
  nc_im = opt.nc_im
  opt.nc_im = 1
  mask = np2torch(mask, opt)
  opt.nc_im = nc_im
  mask = mask.expand(1, 3, mask.shape[2], mask.shape[3])
  plt.imsave('%s/%s_mask_dilated.png' % (
    opt.ref_dir, opt.ref_name[:-4]), convert_image_np(mask), vmin=0, vmax=1)
  mask = (mask - mask.min()) / (mask.max() - mask.min())
  return mask


def add_margin(img, pad):
  width, height = img.size
  new_width = int((1 + 2 * pad) * width)
  new_height = int((1 + 2 * pad) * height)
  result = Image.new(img.mode, (new_width, new_height), (255, 255, 255))
  result.paste(img, (int(pad * width), int(pad * height)))
  return result


def render_text(text, width, height, pad):
  font = ImageFont.truetype('arial.ttf', 28, encoding='unic')
  im = Image.new('RGB', font.getsize(text), (255, 255, 255))
  dctx = ImageDraw.Draw(im)
  dctx.text((0, 0), text, font=font, fill=(0, 0, 0))
  del dctx
  im = add_margin(im, pad)
  im = im.resize((width, height))
  im = np.array(im)
  return im


def generate_text_emb(text_img, resnet, pad=0.05):
  features = []
  def hook_features(module, input, output):
    features.append(output.squeeze())
  handle = resnet._modules['resnet'].layer4.register_forward_hook(hook_features)
  text_img = text_img.float()
  _ = resnet(text_img)
  emb_t = features[0][None, ...]
  handle.remove()
  return emb_t


def rgb2gray(rgb):
  r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
  gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
  return gray[None, ...]


def ocr(roi, width, height, converter, trba_net, opt):
  roi = rgb2gray(roi)
  ac = AlignCollate(imgH=height, imgW=width, keep_ratio_with_pad=opt.pad)
  roi_tensor = ac([roi]).to(opt.device)

  # For max length prediction.
  length_for_pred = torch.IntTensor([opt.batch_max_length]).to(opt.device)
  text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(opt.device)
  preds = trba_net(roi_tensor, text_for_pred, is_train=False)
  return torch.squeeze(preds, 0)


def calc_err_ocr(target_text, fake, converter, trba_net, opt):
  _, c, h, w = fake.shape
  final_text = ocr(fake, w, h, converter, trba_net, opt).to(opt.device)
  target_text_idxs = torch.LongTensor([opt.character.index(c) for c in target_text])
  target_text_idxs = F.pad(target_text_idxs, (0, final_text.shape[0] - len(target_text)))
  target_text_idxs = target_text_idxs.to(opt.device)
  criterion = nn.CrossEntropyLoss()
  err_ocr = opt.lambda_ocr * criterion(final_text, target_text_idxs)
  return err_ocr


def save_losscurve(loss_arr, file_name):
  plt.figure()
  plt.plot(np.arange(len(loss_arr)), loss_arr)
  plt.xlabel('Step')
  plt.ylabel('Loss')
  plt.savefig(file_name)
  plt.close()
