"""SinGAN modules."""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Sequential):
  def __init__(self, in_channel, out_channel, ker_size, padd, stride):
    super(ConvBlock,self).__init__()
    self.add_module('conv', nn.Conv2d(
      in_channel, 
      out_channel,
      kernel_size=ker_size,
      stride=stride,
      padding=padd)),
    self.add_module('norm', nn.BatchNorm2d(out_channel)),
    self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv2d') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('Norm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)
   

class WDiscriminator(nn.Module):
  def __init__(self, opt):
    super(WDiscriminator, self).__init__()
    self.is_cuda = torch.cuda.is_available()
    N = int(opt.nfc)
    self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, 1)
    self.body = nn.Sequential()
    for i in range(opt.num_layer - 2):
      N = int(opt.nfc / pow(2, i + 1))
      block = ConvBlock(
        max(2 * N, opt.min_nfc), 
        max(N, opt.min_nfc), 
        opt.ker_size, 
        opt.padd_size, 
        1)
      self.body.add_module('block%d' % (i + 1), block)
    self.tail = nn.Conv2d(
      max(N, opt.min_nfc), 
      1, 
      kernel_size=opt.ker_size, 
      stride=1, 
      padding=opt.padd_size)

  def forward(self, x):
    # print('D x shape', x.shape)
    x = self.head(x)
    # print('D x shape', x.shape)
    x = self.body(x)
    # print('D x shape', x.shape)
    x = self.tail(x)
    # print('D x shape', x.shape)
    return x


class GeneratorConcatSkip2CleanAdd(nn.Module):
  def __init__(self, opt):
    super(GeneratorConcatSkip2CleanAdd, self).__init__()
    self.is_cuda = torch.cuda.is_available()
    N = opt.nfc
    self.head = ConvBlock(
      opt.nc_z + opt.nc_im,
      N,
      opt.ker_size,
      opt.padd_size,
      1)
    self.body = nn.Sequential()
    for i in range(opt.num_layer - 2):
      N = int(opt.nfc / pow(2, i + 1))
      block = ConvBlock(
        max(2 * N, opt.min_nfc),
        max(N, opt.min_nfc),
        opt.ker_size,
        opt.padd_size,
        1)
      self.body.add_module('block%d' % (i + 1), block)
    self.tail = nn.Sequential(
      nn.Conv2d(
        max(N, opt.min_nfc),
        opt.nc_im,
        kernel_size=opt.ker_size,
        stride=1,
        padding=opt.padd_size),
      nn.Tanh()
    )
  
  def forward(self, x, y):
    # print('G x shape, y shape', x.shape, y.shape)
    x = self.head(x)
    # print('G x shape', x.shape)
    x = self.body(x)
    # print('G x shape', x.shape)
    x = self.tail(x)
    # print('G x shape', x.shape)
    ind = int((y.shape[2] - x.shape[2]) / 2)
    y = y[:, :, ind:(y.shape[2] - ind), ind:(y.shape[3] - ind)]
    # print('G y shape', y.shape)
    return x + y  # torch.cat([x, y], axis=1)


class ResNet34(nn.Module):
  def __init__(self, pretrained=True):
    super().__init__()
    self.resnet = torchvision.models.resnet34(pretrained=pretrained)
  
  def forward(self, x):
    outputs = self.resnet(x)
    return outputs
