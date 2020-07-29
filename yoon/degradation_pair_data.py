import os
import sys

import glob

import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

from img_resize import imresize
from scipy.io import loadmat

def img_random_crop(img, length, yx=None):
    img_size = img.shape
    y_range = img_size[0] - length + 1
    x_range = img_size[1] - length + 1
    if yx:
        y = yx[0]
        x = yx[1]
    else:
        [y, x] = np.random.randint(0, [y_range, x_range])
    return img[y:(y+length), x:(x+length)], [y, x]

class TestDataSR(Dataset):
    def __init__(self, lr_folder, gt_folder=None, test_folder=None, permute=False, bgr2rgb=True):
        super(TestDataSR).__init__()
        self.bgr2rgb = bgr2rgb
        self.permute = permute
        self.lr_folder = lr_folder
        self.lr_files = glob.glob(os.path.join(lr_folder, "**.png"))
        self.lr_files.sort()
        self.gt_folder = None
        if gt_folder:
            self.gt_folder = gt_folder

        if permute:
            np.random.shuffle(self.lr_files)

        self.selected_test_samples = ["0913.png", "0935.png"]
        self.test_samples = []
        self.test_folder = test_folder
        if test_folder:
            for s in self.selected_test_samples:
                afile = os.path.join(test_folder, s)
                img = cv2.imread(afile)
                if bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.test_samples.append(img)

    def get_test_sample(self, index):
        return {"LQ":TF.to_tensor(self.test_samples[index]).unsqueeze(0),
                "LQ_path":[os.path.join(self.test_folder, self.selected_test_samples[index])]}

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, index):
        sample = dict()
        img = cv2.imread(self.lr_files[index])
        if self.bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lr = TF.to_tensor(img)
        sample["LQ"] = lr
        sample["LQ_path"] = self.lr_files[index]
        if self.gt_folder:
            filename = os.path.basename(self.lr_files[index])
            gt = cv2.imread(os.path.join(self.gt_folder, filename))
            if self.bgr2rgb:
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            gt = TF.to_tensor(gt)
            sample["GT"] = gt
            sample["GT_path"] = os.path.join(self.gt_folder, filename)
        return sample

class DegradationParing(Dataset):
    def __init__(self, hr_folder, kern_folder, noise_folder, scale_factor, sr_size, corrupted_folder=None, clean_sc=2, permute=True, bgr2rgb=True):
        super(DegradationParing, self).__init__()
        assert(0 < scale_factor <= 1)
        self.scale_factor = scale_factor
        self.hr_size = sr_size
        self.lr_size = int(sr_size * scale_factor)
        self.bgr2rgb = bgr2rgb # opencv loader
        self.hr_folder = hr_folder
        self.kern_folder = kern_folder
        self.noise_folder = noise_folder
        self.corrupted_folder = corrupted_folder
        self.hr_files = glob.glob(os.path.join(hr_folder, "**.png"))
        self.kern_files = glob.glob(os.path.join(kern_folder, "**/*_x4.mat"))
        self.noise_files = glob.glob(os.path.join(noise_folder, "**.png"))
        self.clean_sc = clean_sc
        if self.corrupted_folder:
            self.corrupted_folder = glob.glob(os.path.join(corrupted_folder, "**.png"))
            self.hr_files.extend(self.corrupted_folder)

        if permute:
            np.random.shuffle(self.hr_files)
            np.random.shuffle(self.kern_files)
            np.random.shuffle(self.noise_files)
    
    def add_noise_and_preproc(self, hr_im, lr_im, noise):
        lr, yx = img_random_crop(lr_im, self.lr_size)
        yx[0] = int(round(yx[0] / self.scale_factor))
        yx[1] = int(round(yx[1] / self.scale_factor))
        hr, _ = img_random_crop(hr_im, self.hr_size, yx)
        z_im, _ = img_random_crop(noise, self.lr_size)

        if random.random() < 0.5:
            z_im = cv2.flip(z_im, 0) # vertical
        if random.random() < 0.5:
            z_im = cv2.flip(z_im, 1) # horizontal
        z_mean = np.mean(z_im.reshape(-1, 3), 0).reshape((1, 1, 3))
        z = z_im.astype(np.float) - z_mean
        lr = np.clip(np.round(lr.astype(np.float) + z), 0, 255).astype(np.uint8) # add noise

        if random.random() < 0.5:
            hr = cv2.flip(hr, 0) # vertical flip
            lr = cv2.flip(lr, 0)

        if random.random() < 0.5:
            hr = cv2.flip(hr, 1) # horizontal
            lr = cv2.flip(lr, 1)

        # cv2.imshow("hr", hr[:,:,::-1])
        # cv2.imshow("lr", lr[:,:,::-1])
        # cv2.imshow("z", z_im[:,:,::-1])

        hr = TF.to_tensor(hr)
        lr = TF.to_tensor(lr)
        return hr, lr, z_im

    def __getitem__(self, index):
        kernel = np.array(loadmat(self.kern_files[index])["Kernel"])
        rand_idx = np.random.randint(0, [len(self.hr_files), len(self.noise_files)])
        hr_file = self.hr_files[rand_idx[0]]
        hr_img = cv2.imread(hr_file)
        if os.path.basename(hr_file)[:8] == "Flickr2K": # hr_file in self.corrupted_folder:
            hr_img = cv2.resize(hr_img, dsize=(0, 0), fx=1./self.clean_sc, fy=1./self.clean_sc, interpolation=cv2.INTER_AREA)
        noise = cv2.imread(self.noise_files[rand_idx[1]])
        hr_img, _ = img_random_crop(hr_img, int(self.hr_size * 2))
        if self.bgr2rgb:
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB) # hr_img[:, :, [2, 1, 0]]
            noise = cv2.cvtColor(noise, cv2.COLOR_BGR2RGB) # noise[:, :, ::-1]
        dwn_img = imresize(hr_img, self.scale_factor, kernel=kernel)
        hr, lr, z = self.add_noise_and_preproc(hr_img, dwn_img, noise)
        # cv2.imshow("hr_ori", hr_img[:,:,::-1])
        # cv2.imshow("lr_ori", dwn_img[:,:,::-1])
        # cv2.waitKey()
        return {"GT":hr, "LQ":lr, "z":z, "LQ_path":self.hr_files[rand_idx[0]], "GT_path":self.hr_files[rand_idx[0]]}

    def __len__(self):
        return len(self.kern_files)

def valid():
    lr_folder = "/mnt/data/NTIRE2020/realSR/track1/Corrupted-va-x"
    gt_folder = "/mnt/data/NTIRE2020/realSR/track1/DIV2K_valid_HR"
    test_folder = "/mnt/data/NTIRE2020/realSR/track1/Corrupted-te-x"
    data = TestDataSR(lr_folder, gt_folder, test_folder)

    n_test = len(data.test_samples)
    for t in range(n_test):
        t_im = data.get_test_sample(t)
        lr = t_im["LQ"][0].cpu().numpy().transpose(1, 2, 0) * 255
        lr = np.round(lr).astype(np.uint8)
        cv2.imshow("te_{}".format(t), lr)

    for i in range(len(data)):
        lr = data[i]["LQ"].cpu().numpy().transpose(1, 2, 0) * 255
        lr = np.round(lr).astype(np.uint8)
        gt = data[i]["GT"].cpu().numpy().transpose(1, 2, 0) * 255
        gt = np.round(gt).astype(np.uint8)
        cv2.imshow("lr", lr[:, :, ::-1])
        cv2.imshow("gt", gt[:, :, ::-1])
    cv2.waitKey()

if __name__ == "__main__":
    #valid()
    seed_num = 0
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_num)
    random.seed(seed_num)

    hr_folder = "/mnt/data/NTIRE2020/realSR/track1/Corrupted-tr-y"
    kern_folder = "yoon/kernels/track1"
    noise_folder = "yoon/noises/track1/p128_v100"
    corrupted_folder = "/mnt/data/NTIRE2020/realSR/track1/Corrupted-tr-x"

    data = DegradationParing(hr_folder, kern_folder, noise_folder, 0.25, 64, corrupted_folder=corrupted_folder, clean_sc=2)
    loader = DataLoader(data, batch_size=4, shuffle=True)
    for i, D in enumerate(loader):
        for j in range(len(D["GT"])):
            hr = D["GT"][j].cpu().numpy().transpose(1, 2, 0) * 255
            lr = D["LQ"][j].cpu().numpy().transpose(1, 2, 0) * 255
            hr = np.round(hr).astype(np.uint8)
            lr = np.round(lr).astype(np.uint8)
            cv2.imshow("hr_{}".format(j), hr[:, :, ::-1])
            cv2.imshow("lr_{}".format(j), lr[:, :, ::-1])
            cv2.imshow("z_{}".format(j), D["z"][j].cpu().numpy())
        cv2.waitKey()
    print("fin.")
