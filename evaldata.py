import numpy as np
from PIL import Image
import argparse
from skimage.color import rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from getDatasetPath import getDatasetPath

def RMSEinLAB(img1, img2, mask):
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)
    diff2 = np.power(img1_lab - img2_lab, 2)

    if mask.ndim == 2:
        mask = np.tile(mask, (3, 1, 1)).transpose(1, 2, 0)

    shadow_rmse = np.sqrt((diff2 * mask).sum(axis=(0, 1)) / mask[:, :, 0].sum())
    nonshadow_rmse = np.sqrt((diff2 * (1.0 - mask)).sum(axis=(0, 1)) / (1.0 - mask[:, :, 0]).sum())
    whole_rmse = np.sqrt(diff2.mean(axis=(0, 1)))

    return shadow_rmse.sum(), nonshadow_rmse.sum(), whole_rmse.sum()


def MAEinLAB(img1, img2, mask):
    img1_lab = rgb2lab(img1)
    img2_lab = rgb2lab(img2)
    diff = np.abs(img1_lab - img2_lab)

    if mask.ndim == 2:
        mask = np.tile(mask, (3, 1, 1)).transpose(1, 2, 0)

    shadow_mae = (diff * mask).sum(axis=(0, 1)) / mask[:, :, 0].sum()
    nonshadow_mae = (diff * (1.0 - mask)).sum(axis=(0, 1)) / (1.0 - mask[:, :, 0]).sum()
    whole_mae = diff.mean(axis=(0, 1))

    return np.sum(shadow_mae), np.sum(nonshadow_mae), np.sum(whole_mae)


def PSNRinsRGB(img1, img2):

    return psnr(img1, img2, data_range=1.0)


def SSIMinsRGB(img1, img2):

    return ssim(img1, img2, data_range=1.0, multichannel=True)

# ------- setting --------
parser = argparse.ArgumentParser()
parser.add_argument("--method_name", type=str, default="CANet", help="Folder name to evaluate")
parser.add_argument("--dataset_name", type=str, default="ISTD", help="Folder Name of the dataset")
parser.add_argument("--resized", type=int, default=1, help="0: not resized, 1: resized to 256x256")
opt = parser.parse_args()
# -------------------------

output_img_path, gt_img_path, mask_img_path = getDatasetPath(opt.method_name, opt.dataset_name)

if (output_img_path == 0):
    exit()

RMSEresult = []
MAEresult = []
PSNRresult = []
SSIMresult = []
for i in range(len(output_img_path)):
    output_img = Image.open(output_img_path[i])
    gt_img = Image.open(gt_img_path[i])
    mask_img = Image.open(mask_img_path[i])

    if opt.resized == 0:  # Not resized
        neww = min(output_img.size[0], gt_img.size[0])
        newh = min(output_img.size[1], gt_img.size[1])
    else:  # Resized to 256 x 256
        neww = 256
        newh = 256

    output_img = output_img.resize((neww, newh), Image.NEAREST)
    gt_img = gt_img.resize((neww, newh), Image.NEAREST)
    mask_img = mask_img.resize((neww, newh), Image.NEAREST)

    output_img = np.array(output_img, 'f') / 255.
    gt_img = np.array(gt_img, 'f') / 255.
    mask_img = (np.array(mask_img, dtype=np.int32) / 255).astype(np.float32)

    if mask_img.sum() == 0:
        print("skipped {}".format(output_img_path[i]))
        continue

    RMSEresult.append(RMSEinLAB(output_img, gt_img, mask_img))
    MAEresult.append(MAEinLAB(output_img, gt_img, mask_img))
    PSNRresult.append(PSNRinsRGB(output_img, gt_img))
    SSIMresult.append(SSIMinsRGB(output_img, gt_img))

print("=========")
print("Dataset: {} / Method: {} / Resize : {}".format(opt.dataset_name, opt.method_name, "No" if opt.resized == 0 else "256x256"))
print("== RMSE ==")
print("shadow: {0[0]:.2f}, Non-shadow:{0[1]:.2f}, All: {0[2]:.2f}".format(np.array(RMSEresult).mean(0)))
print("== MAE ==")
print("shadow: {0[0]:.2f}, Non-shadow:{0[1]:.2f}, All: {0[2]:.2f}".format(np.array(MAEresult).mean(0)))
print("== PSNR ==")
print("All: {0:.2f}".format(np.array(PSNRresult).mean()))
print("== SSIM ==")
print("All: {0:.3f}".format(np.array(SSIMresult).mean()))
print("=========")
