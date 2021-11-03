"""The main file to run training."""

from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from nltk.corpus import words
from trba import TRBA
from trba_utils import AttnLabelConverter
import cv2


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

  # Load the weights from the pre-trained text recognition model.
  converter = AttnLabelConverter(opt.character)
  opt.num_class = len(converter.character)

  if opt.use_rgb:
    opt.input_channel = 3
  trba_net = TRBA(opt).to(opt.device)

  print('[INFO] loading TRBA text recognition model...')
  model_dict = torch.load(opt.trba, map_location=opt.device)
  # Remove "module." from key names because we are not using DataParallel.
  model_dict = {'.'.join(k.split('.')[1:]): v for k, v in model_dict.items()}
  trba_net.load_state_dict(model_dict)
  trba_net.eval()

  # if (os.path.exists(dir2save)):
  #   print('Trained model already exists.')
  # else:
  try:
    os.makedirs(dir2save)
  except OSError:
    pass

  # Get source image.
  source_img = cv2.imread(f'Images/{opt.input_name}')
  height, width, _ = source_img.shape

  real = functions.read_image(opt)
  functions.adjust_scales2image(real, opt)
  resnet = models.ResNet34()
  emb_fixed = functions.generate_text_emb(opt.text, width, height, resnet)
  train(opt, Gs, words.words(), converter, trba_net, resnet, emb_fixed, height, width, reals, NoiseAmp) # opt, Gs, word_bank, resnet, emb_fixed, reals, NoiseAmp
  # SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt)
