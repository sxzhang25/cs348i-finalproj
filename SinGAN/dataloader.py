from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from SinGAN.functions import render_text


class WordDataset(Dataset):
  def __init__(self, word_bank, character, width, height, pad):
    self.word_bank = word_bank
    self.width = width
    self.height = height
    self.pad = pad
    self.character = character

  def __len__(self):
    return len(self.word_bank)

  def __getitem__(self, idx):
    word = self.word_bank[idx]
    word = ''.join([c for c in word if c in self.character])
    X = render_text(word, self.width, self.height, self.pad)
    X = np.transpose(X, [2, 0, 1])
    return X, word


def create_dataset(word_bank, character, width, height, pad, B=1, shuffle=True):
  dset = WordDataset(word_bank, character, width, height, pad)
  loader = DataLoader(dset, batch_size=B, shuffle=shuffle)
  return loader
