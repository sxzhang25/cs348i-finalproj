"""The main file to run training."""

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from SinGAN.dataloader import create_dataset
from nltk.corpus import words
from trba import TRBA
from trba_utils import AttnLabelConverter
import cv2
import torch


if __name__ == '__main__':
  parser = get_arguments()
  parser.add_argument(
    '--input_dir', default='Images',
    help='Input image directory.')
  parser.add_argument(
    '--text', type=str, required=True, 
    help='Text in source image.')
  parser.add_argument(
    '--input_name', required=True,
    help='Input image name.')
  parser.add_argument(
    '--mode', default='train',
    help='Task to be done.')

  opt = parser.parse_args()
  opt = functions.post_config(opt)
  Gs = []
  Zs = []
  reals = []
  NoiseAmp = []
  dir2save = functions.generate_dir2save(opt)

  if opt.use_resnet:
    opt.nc_z = 512

  # Load the weights from the pre-trained text recognition model.
  converter = AttnLabelConverter(opt.character)
  opt.num_class = len(converter.character)

  print('[INFO] loading TRBA text recognition model...', flush=True)
  trba_net = TRBA(opt).to(opt.device)
  model_dict = torch.load(opt.trba, map_location=opt.device)
  # Remove "module." from key names because we are not using DataParallel.
  model_dict = {'.'.join(k.split('.')[1:]): v for k, v in model_dict.items()}
  trba_net.load_state_dict(model_dict)

  try:
    os.makedirs(dir2save)
  except OSError:
    pass

  # Get source image.
  source_img = cv2.imread(f'{opt.input_dir}/{opt.input_name}')
  height, width, _ = source_img.shape

  real = functions.read_image(opt)
  functions.adjust_scales2image(real, opt)
  resnet = models.ResNet34()
  
  fixed_text_img = functions.render_text(opt.text, width, height, opt.pad)[None, ...]
  fixed_text_img = torch.Tensor(np.transpose(fixed_text_img, [0, 3, 1, 2]))
  if opt.use_resnet:
    emb_fixed = functions.generate_text_emb(fixed_text_img, resnet)
  else:
    emb_fixed = fixed_text_img
  
  # Create word dataloader.
  word_bank = create_dataset(words.words(), opt.character, width, height, opt.pad)

  # Print configs.
  print('CONFIGS:', opt, '\n')

  # Start training.
  train(opt, Gs, word_bank, converter, trba_net, resnet, emb_fixed, height, width, reals, NoiseAmp) # opt, Gs, word_bank, resnet, emb_fixed, reals, NoiseAmp
  # SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
