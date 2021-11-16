"""Configs for training a SinGAN."""

import argparse

def get_arguments():
  parser = argparse.ArgumentParser()
  # Workspace args.
  parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    
  # Load, input, save configurations.
  parser.add_argument(
    '--netG', default='', 
    help='Path to netG (to continue training).')
  parser.add_argument(
    '--netD', default='', 
    help='Path to netD (to continue training).')
  parser.add_argument(
    '--manualSeed', type=int, 
    help='Manual seed.')
  parser.add_argument(
    '--nc_z',type=int, default=3,
    help='Noise # channels.')
  parser.add_argument(
    '--nc_im', type=int, default=3,
    help='Image # channels.')
  parser.add_argument(
    '--out', default='Output',
    help='Output folder.')
  parser.add_argument(
    '--use_resnet', action='store_true', 
    help='If true, use ResNet embeddings as input. Otherwise, use rendered plaintext.')
  parser.add_argument(
    '--save_freq', type=int, default=25, 
    help='Save every k steps.')
        
  # Networks hyper parameters.
  parser.add_argument(
    '--nfc', type=int, default=32)
  parser.add_argument(
    '--min_nfc', type=int, default=32)
  parser.add_argument(
    '--ker_size', type=int, default=3,
    help='Kernel size.')
  parser.add_argument(
    '--num_layer', type=int, default=5,
    help='Number of layers.')
  parser.add_argument(
    '--stride', default=1)
  parser.add_argument(
    '--padd_size', type=int, default=0,
    help='Net pad size.')
        
  # Pyramid parameters.
  parser.add_argument(
    '--patch_scale', type=int, default=1, 
    help='The scale at which to start enforcing patch discrimination loss.')
  parser.add_argument(
    '--scale_factor', type=float, default=0.75,
    help='Pyramid scale factor.')
  parser.add_argument(
    '--noise_amp', type=float, default=0.1,
    help='Additive noise cont weight.')
  parser.add_argument(
    '--min_size', type=int, default=25,
    help='Image minimal size at the coarser scale.')
  parser.add_argument(
    '--max_size', type=int, default=250,
    help='Image minimal size at the coarser scale.')

  # Optimization hyper parameters.
  parser.add_argument(
    '--niter', type=int, default=2000, 
    help='number of epochs to train per scale')
  parser.add_argument(
    '--gamma', type=float, default=0.1,
    help='Scheduler gamma.')
  parser.add_argument(
    '--lr_g', type=float, default=0.0005, 
    help='learning rate, default=0.0005')
  parser.add_argument(
    '--lr_d', type=float, default=0.0005, 
    help='learning rate, default=0.0005')
  parser.add_argument(
    '--beta1', type=float, default=0.5, 
    help='beta1 for adam. default=0.5')
  parser.add_argument(
    '--Gsteps', type=int, default=3,
    help='Generator inner steps')
  parser.add_argument(
    '--Dsteps', type=int, default=3,
    help='Discriminator inner steps')
  parser.add_argument(
    '--lambda_grad', type=float, default=0.1,
    help='gradient penalty weight')
  parser.add_argument(
    '--alpha', type=float, default=10,
    help='reconstruction loss weight')
  parser.add_argument(
    '--lambda_ocr', type=float, default=0.1, 
    help='text recognition loss')

  # TRBA text recognition args.
  parser.add_argument(
    '--trba', default='models/TPS-ResNet-BiLSTM-Attn.pth', type=str, 
    help='Path to TRBA model.')
  parser.add_argument(
    '--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', 
    help='Character label.')
  parser.add_argument(
    '--batch_max_length', type=int, default=25, 
    help='Maximum label length.')
  parser.add_argument(
    '--imgH', type=int, default=32, 
    help='The height of the input image.')
  parser.add_argument(
    '--imgW', type=int, default=100, 
    help='The width of the input image.')
  parser.add_argument(
    '--pad', action='store_true', 
    help='Whether to keep ratio then pad for image resize.')
  parser.add_argument(
    '--num_fiducial', type=int, default=20, 
    help='Number of fiducial points of TPS-STN.')
  parser.add_argument(
    '--input_channel', type=int, default=1, 
    help='The number of input channels in the feature extractor.')
  parser.add_argument(
    '--output_channel', type=int, default=512,
    help='The number of output channel in the feature extractor.')
  parser.add_argument(
    '--hidden_size', type=int, default=256, 
    help='The size of the LSTM hidden state.')
    
  return parser
