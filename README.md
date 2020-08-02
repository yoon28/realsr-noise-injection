# Intro.

This repo is an unofficial implementation of [RealSR](https://github.com/jixiaozhong/RealSR) including training codes and kernel/noise estimation codes.

# Usage

## kernel and noise estimation

You can download the estimated kernel from [here](https://sianalytics-my.sharepoint.com/:u:/g/personal/yoon28_si-analytics_ai/EVr8bcSAy4lIlTA14L7TClMBmfjQop-MpnM4Y_XFgKqoWA?e=kO4uyB) and the noise images from [here](https://sianalytics-my.sharepoint.com/:u:/g/personal/yoon28_si-analytics_ai/EVm7nV1BekFJtd6xr75sEJQBhY2FdQqIW1o3bZkVv4DEtA?e=ZhhqEB).

Furthermore, you can estimate kernels with NTIRE2020 real-world SR data by executing `stage1_kernel.py` located under `yoon` folder.
Since the kernel estiamtion needs [KernelGAN](https://github.com/sefibk/KernelGAN), you have to designate the location of KernelGAN codes, eg. `export PYTHONPATH=/path/to/kernelgan`.

For the noises, you can also use `stage1_noise.py` to estimate noises from corrupted images rather than downloading the noise images from the above.
Note that the noises you downloaded are produced with settings `patch_size=128` and `max_var=100`.

To locate the NTIRE2020 dataset, you can modify the variables `DATA_LOC`, `DATA_X`, `DATA_Y`, `DATA_VAL` that are located at the early part of `stage1_kernel.py` and `stage1_noise.py`.

## training

For training, please use `train_realsr.py` codes located under `yoon` folder.
For example:

```
PYTHONPATH=/mnt/workspace/SR/RealSR/codes CUDA_VISIBLE_DEVICES=14,15 python3 yoon/train_realsr.py -opt yoon/options/train_df2k.yml
```

This example first registers python codes under `codes` folders to `PYTHONPATH` since I separated original codes from the codes I implemented. And it declares 2 gpus for the training procedure.

# Misc.

First, this repo is somewhat messy because I implemented the paper for my own needs. Second, I fail to reproduce the results in the sense of SR quality. With my implementation, I saw that noises are very well removed but the sharpness of my results is not as much good as the original results.

In my opinion, reproducing the paper is difficult because the author did not share many of the hyper-paramter settings, such as the variance cutoff (`max_var`), the size of noise patch (`patch_size`), the clean-up scale factor and so on.

# Reference

Ji, Xiaozhong, et al. "Real-World Super-Resolution via Kernel Estimation and Noise Injection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops. 2020.