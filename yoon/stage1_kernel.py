import os, sys

import numpy as np
import cv2
import random
import torch

from configs import Config
from kernelGAN import KernelGAN
from data import DataGenerator
from learner import Learner

import tqdm

DATA_LOC = "/mnt/data/NTIRE2020/realSR/track2" # "/mnt/data/NTIRE2020/realSR/track1"
DATA_X = "DPEDiphone-tr-x" # "Corrupted-tr-x"
DATA_Y = "DPEDiphone-tr-y" # "Corrupted-tr-y"
DATA_VAL = "DPEDiphone-va" # "Corrupted-va-x"

def config_kernelGAN(afile):
    img_folder = os.path.dirname(afile)
    img_file = os.path.basename(afile)
    out_dir = "yoon/kernels/track2"

    params = ["--input_image_path", afile,
            "--output_dir_path", out_dir,
            "--noise_scale", str(1.0),
            "--X4"]
    conf = Config().parse(params)
    conf.input2 = None
    return conf

def estimate_kernel(img_file):
    conf = config_kernelGAN(img_file)
    kgan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, kgan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=70):
        [g_in, d_in, _] = data.__getitem__(iteration)
        kgan.train(g_in, d_in)
        learner.update(iteration, kgan)
    kgan.finish()

if __name__ == "__main__":
    seed_num = 0
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_num)
    random.seed(seed_num)

    # exit(0)

    data = {"X":[os.path.join(DATA_LOC, DATA_X, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_X)) if f[-4:] == ".png"],
            "Y":[os.path.join(DATA_LOC, DATA_Y, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_Y)) if f[-4:] == ".png"],
            "val":[os.path.join(DATA_LOC, DATA_VAL, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_VAL)) if f[-4:] == ".png"]}

    Kernels = []
    Noises = []
    for f in data["X"]:
        estimate_kernel(f)
    print("fin.")
