# cs348i-finalproj

Based off of the paper "SinGAN: Learning a Generative Model from a Single Natural Image" in ICCV 2019.

Official paper repository: https://github.com/tamarott/SinGAN

## Pretrained models

In order to run training, first download [this model](https://drive.google.com/open?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY) and save it in a folder called `models/`.

## Train
To train SinGAN model on your own image, put the desired training image under `Images/`, and run

```
python main_train.py --input_name <input_file_name>
```

This will also use the resulting trained model to generate random samples starting from the coarsest scale (n = 0).

To run this code on a cpu machine, specify `--not_cuda` when calling `main_train.py`.