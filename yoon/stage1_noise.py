import os, sys

import numpy as np
import cv2
import random
import torch

DATA_LOC = "/mnt/data/NTIRE2020/realSR/track2" #  "/mnt/data/NTIRE2020/realSR/track1"
DATA_X = "DPEDiphone-tr-x" # "Corrupted-tr-x" #
DATA_Y = "DPEDiphone-tr-y" # "Corrupted-tr-y" #
DATA_VAL = "DPEDiphone-va" # "Corrupted-va-x"
OUT_DIR = "yoon/noises/track2"

def noises_estimation(img, rescale_0_1=False):
    patch_size = 64
    block_size = 16
    stride_g = patch_size // 2
    stride_l = block_size
    mu = 0.1
    lambd = 0.25
    noises = []
    im_size = img.shape[:2]
    for y in range(0, im_size[0], stride_g):
        for x in range(0, im_size[1], stride_g):
            if x + patch_size > im_size[1] or y + patch_size > im_size[0]:
                continue
            if rescale_0_1:
                patch = img[y:(y+patch_size), x:(x+patch_size), :].astype(np.float) / 255.0
            else:
                patch = img[y:(y+patch_size), x:(x+patch_size), :].astype(np.float)
            mean_patch = np.mean(patch.reshape((-1, 3)), 0)
            var_patch = np.var(patch.reshape((-1, 3)), 0)

            is_noise = True
            for j in range(0, patch_size, stride_l):
                for i in range(0, patch_size, stride_l):
                    if j+block_size > patch_size or i+block_size > patch_size:
                        continue
                    block = patch[j:(j+block_size), i:(i+block_size), :]
                    #assert block.shape[0] == block_size and block.shape[1] == block_size
                    mean_block = np.mean(block.reshape((-1, 3)), 0)
                    var_block = np.var(block.reshape((-1, 3)), 0)
                    if np.greater(np.abs(mean_block - mean_patch), mean_patch*mu).any():
                        is_noise = False
                        break
                    if np.greater(np.abs(var_block - var_patch), var_patch*lambd).any():
                        is_noise = False
                        break
                if not is_noise:
                    break
            if is_noise:
                noises.append(img[y:(y+patch_size), x:(x+patch_size), :])
    return noises

def noises_estimation_simple(img, rescale_0_1=False):
    patch_size = 128
    stride_g = patch_size // 2
    max_var = (10*10)
    min_var = (0*0)
    noises = []
    im_size = img.shape[:2]
    for y in range(0, im_size[0], stride_g):
        for x in range(0, im_size[1], stride_g):
            if x + patch_size > im_size[1] or y + patch_size > im_size[0]:
                continue
            if rescale_0_1:
                patch = img[y:(y+patch_size), x:(x+patch_size), :].astype(np.float) / 255.0
            else:
                patch = img[y:(y+patch_size), x:(x+patch_size), :].astype(np.float)
            var_patch = np.var(patch.reshape((-1, 3)), 0)
            if np.less(var_patch, max_var).all() and np.greater(var_patch, min_var).all():
                noises.append(img[y:(y+patch_size), x:(x+patch_size), :])
    return noises

if __name__ == "__main__":
    seed_num = 0
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_num)
    random.seed(seed_num)

    exit(0)

    data = {"X":[os.path.join(DATA_LOC, DATA_X, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_X)) if f[-4:] == ".png"],
            "Y":[os.path.join(DATA_LOC, DATA_Y, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_Y)) if f[-4:] == ".png"],
            "val":[os.path.join(DATA_LOC, DATA_VAL, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_VAL)) if f[-4:] == ".png"]}

    out_dir = os.path.join(OUT_DIR, "p128_v100")
    j = 0
    for f in data["X"]:
        print(f)
        img = cv2.imread(f)
        N = noises_estimation_simple(img)
        for n in N:
            j += 1
            filename = os.path.join(out_dir, "noise_{:08d}.png".format(j))
            cv2.imwrite(filename, n)
            print("\tsaved: ", filename)

# if __name__ == "__main__":
#     seed_num = 0
#     torch.manual_seed(seed_num)
#     torch.cuda.manual_seed(seed_num)
#     torch.cuda.manual_seed_all(seed_num)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed_num)
#     random.seed(seed_num)

#     data = {"X":[os.path.join(DATA_LOC, DATA_X, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_X)) if f[-4:] == ".png"],
#             "Y":[os.path.join(DATA_LOC, DATA_Y, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_Y)) if f[-4:] == ".png"],
#             "val":[os.path.join(DATA_LOC, DATA_VAL, f) for f in os.listdir(os.path.join(DATA_LOC, DATA_VAL)) if f[-4:] == ".png"]}
#     print(cv2.__version__)
#     Noises = []
#     for j, f in enumerate(data["X"]):
#         img = cv2.imread(f)
#         N = noises_estimation_simple(img)
#         Noises.extend(N)
#         cv2.imshow("X", img)
#         for i, n in enumerate(N):
#             cv2.imshow("N_{}".format(i), n)
#             cv2.waitKey()
#         cv2.destroyAllWindows()
#         print(len(Noises))
#         # if len(Noises) > 5:
#         #     break

#     test_im = cv2.imread(data["Y"][0])
#     im_size = test_im.shape[:2]
#     for i, n in enumerate(Noises):
#         test_canv = test_im.astype(np.float) / 255.0
#         N = n.astype(np.float) / 255.0
#         N_mean = np.mean(N.reshape((-1, 3)), 0).reshape((1, 1, 3))
#         N_size = N.shape[:2]
#         z = (N-N_mean)
#         crop = test_canv[im_size[0]//2:im_size[0]//2+N_size[0], im_size[1]//2:im_size[1]//2+N_size[1], :]
#         corruped = crop + z
#         corruped = np.round((np.clip(corruped, 0, 1) * 255)).astype(np.uint8)
#         crop = (crop * 255).astype(np.uint8)
#         z = np.round(np.clip(z, 0, 1) * 255).astype(np.uint8)
#         cv2.imshow("noise", z)
#         cv2.imshow("noise_patch", n)
#         cv2.imshow("crop", crop)
#         cv2.imshow("orig", test_im)
#         cv2.imshow("corrup", corruped)
#         cv2.waitKey()
#         print(i)
#     print("fin.")
