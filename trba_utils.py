"""TRBA utils.

Referenced from:
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/utils.py
"""

from PIL import Image
import numpy as np
import math
import torch
import torchvision.transforms as T
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CTCLabelConverter(object):
  """Convert between text-label and text-index."""

  def __init__(self, character):
    # character (str): set of the possible characters.
    dict_character = list(character)

    self.dict = {}
    for i, char in enumerate(dict_character):
      # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss.
      self.dict[char] = i + 1

    self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

  def encode(self, text, batch_max_length=25):
    """Convert text-label into text-index.

    Args:
      text: text Labels of each image. [batch_size]
      batch_max_length: Max length of text label in the batch. 25 by default.

    Returns:
      text: Text index for CTCLoss. [batch_size, batch_max_length]
      length: Length of each text. [batch_size]
    """
    length = [len(s) for s in text]

    # The index used for padding (=0) would not affect the CTC loss calculation.
    batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
    for i, t in enumerate(text):
      text = list(t)
      text = [self.dict[char] for char in text]
      batch_text[i][:len(text)] = torch.LongTensor(text)

    return (batch_text.to(device), torch.IntTensor(length).to(device))

  def decode(self, text_index, length):
      """Convert text-index into text-label."""
      texts = []
      for index, l in enumerate(length):
        t = text_index[index, :]

        char_list = []
        for i in range(l):
          if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # Removing repeated characters and blank.
            char_list.append(self.character[t[i]])
        text = ''.join(char_list)
        texts.append(text)

      return texts


class CTCLabelConverterForBaiduWarpctc(object):
  """Convert between text-label and text-index for baidu warpctc."""

  def __init__(self, character):
    # character (str): set of the possible characters.
    dict_character = list(character)

    self.dict = {}
    for i, char in enumerate(dict_character):
      # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss.
      self.dict[char] = i + 1
    self.character = ['[CTCblank]'] + dict_character  # Dummy '[CTCblank]' token for CTCLoss (index 0).

    def encode(self, text, batch_max_length=25):
      """Convert text-label into text-index.

      Args:
        text: Text labels of each image. [batch_size]

      Returns:
        text: Concatenated text index for CTCLoss.
          [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
        length: Length of each text. [batch_size]
      """
      length = [len(s) for s in text]
      text = ''.join(text)
      text = [self.dict[char] for char in text]

      return (torch.IntTensor(text), torch.IntTensor(length))

  def decode(self, text_index, length):
    """Convert text-index into text-label."""
    texts = []
    index = 0
    for l in length:
      t = text_index[index:index + l]

      char_list = []
      for i in range(l):
        if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # Removing repeated characters and blank.
          char_list.append(self.character[t[i]])
      text = ''.join(char_list)

      texts.append(text)
      index += l
    return texts


class AttnLabelConverter(object):
  """Convert between text-label and text-index."""

  def __init__(self, character):
    # character (str): set of the possible characters.
    # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
    list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
    list_character = list(character)
    self.character = list_token + list_character

    self.dict = {}
    for i, char in enumerate(self.character):
      self.dict[char] = i

  def encode(self, text, batch_max_length=25):
    """Convert text-label into text-index.

    Args:
      text: Text labels of each image. [batch_size]
      batch_max_length: Max length of text label in the batch. 25 by default

    Return:
      text: The input of attention decoder. [batch_size x (max_length+2)] +1 
        for [GO] token and +1 for [s] token. text[:, 0] is [GO] token and text
        is padded with [GO] token after [s] token.
      length: the length of output of attention decoder, which also count [s] 
        token. [3, 7, ....] [batch_size]
    """
    length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
    # batch_max_length = max(length) # this is not allowed for multi-gpu setting
    batch_max_length += 1
    # Additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
    batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
    for i, t in enumerate(text):
      text = list(t)
      text.append('[s]')
      text = [self.dict[char] for char in text]
      batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
    return (batch_text.to(device), torch.IntTensor(length).to(device))

  def decode(self, text_index, length):
    """Convert text-index into text-label."""
    texts = []
    for index, l in enumerate(length):
      text = ''.join([self.character[i] for i in text_index[index, :]])
      texts.append(text)
    return texts


class Averager(object):
  """Compute average for torch.Tensor, used for loss average."""

  def __init__(self):
    self.reset()

  def add(self, v):
    count = v.data.numel()
    v = v.data.sum()
    self.n_count += count
    self.sum += v

  def reset(self):
    self.n_count = 0
    self.sum = 0

  def val(self):
    res = 0
    if self.n_count != 0:
      res = self.sum / float(self.n_count)
    return res


class ResizeNormalize(object):

  def __init__(self, size, interpolation='bilinear'):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = T.ToTensor()

  def __call__(self, img):
    # print('img shape', img.shape)
    img = F.interpolate(img, self.size, mode=self.interpolation)
    img.sub_(0.5).div_(0.5)
    return img


class NormalizePad(object):

  def __init__(self, max_size, PAD_type='right'):
    self.toTensor = T.ToTensor()
    self.max_size = max_size
    self.max_width_half = math.floor(max_size[2] / 2)
    self.PAD_type = PAD_type

  def __call__(self, img):
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    c, h, w = img.size()
    Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
    Pad_img[:, :, :w] = img  # Right pad.
    if self.max_size[2] != w:  # Add border pad.
      Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
    return Pad_img


class AlignCollate(object):

  def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
      self.imgH = imgH
      self.imgW = imgW
      self.keep_ratio_with_pad = keep_ratio_with_pad

  def __call__(self, images):
    if self.keep_ratio_with_pad:  # Same concept as 'Rosetta' paper.
      resized_max_w = self.imgW
      input_channel = 3 if images[0].mode == 'RGB' else 1
      transform = NormalizePad((input_channel, self.imgH, resized_max_w))

      resized_images = []
      for image in images:
        w, h = image.size
        ratio = w / float(h)
        if math.ceil(self.imgH * ratio) > self.imgW:
          resized_w = self.imgW
        else:
          resized_w = math.ceil(self.imgH * ratio)

        resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
        resized_images.append(transform(resized_image))

      image_tensors = torch.cat(resized_images, 0)
    else:
      transform = ResizeNormalize((self.imgW, self.imgH))
      image_tensors = [transform(image) for image in images]
      image_tensors = torch.cat(image_tensors, 0)

    return image_tensors