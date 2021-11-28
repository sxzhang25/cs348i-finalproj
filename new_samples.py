from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions


if __name__ == '__main__':
  parser = get_arguments()
  parser.add_argument(
    '--input_dir', help='input image dir', default='Images')
  parser.add_argument(
    '--input_name', help='input image name', required=True)
  parser.add_argument(
    '--text', type=str, required=True, 
    help='The text string to render.')
  parser.add_argument(
    '--mode', type=str, choices=['train', 'new_samples'], default='new_samples')

  opt = parser.parse_args()
  opt = functions.post_config(opt)
  Gs = []
  Zs = []
  reals = []
  NoiseAmp = []
  dir2save = functions.generate_dir2save(opt)

  if dir2save is None:
    print('task does not exist')
  else:
    try:
      os.makedirs(dir2save)
    except OSError:
      pass
    if opt.mode == 'new_samples':
      real = functions.read_image(opt)
      _, height, width = real[0].shape
      functions.adjust_scales2image(real, opt)
      Gs, reals = functions.load_trained_pyramid(opt)
      print(len(Gs), len(reals))
      SinGAN_generate(opt.text, width, height, Gs, reals, opt)



