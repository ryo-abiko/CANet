'''
Loading Dataset
'''

import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf


def srgb2rgb(tensor):
    input_sRGB = tensor.detach().clone()
    input_sRGB[input_sRGB <= 0.04045] /= 12.92
    input_sRGB[input_sRGB > 0.04045] = pow((input_sRGB[input_sRGB > 0.04045] + 0.055) / 1.055, 2.4)

    return input_sRGB.detach()


def rgb2srgb(tensor):
    RGB = tensor.detach().clone()
    RGB[RGB > 0.0031308] = pow(RGB[RGB > 0.0031308], 1 / 2.4) * 1.055 - 0.055
    RGB[RGB <= 0.0031308] *= 12.92

    return RGB.detach()


def rotflipImage(image, rot, flipv, fliph):

    if flipv > 0:
        image = tf.vflip(image)
    if fliph > 0:
        image = tf.hflip(image)

    return tf.rotate(image, rot * 90)


def cropAndTotensor(image, top, left):

    totensor = transforms.ToTensor()

    image = totensor(image)
    image = tf.resize(image, (512, 512), transforms.InterpolationMode.NEAREST)
    image = tf.crop(image, top, left, 256, 256)
    return image


class ImageDataset(Dataset):
    def __init__(self, hr_shape, shadow_dir_path, gt_dir_path, mask_dir_path, mask_edge_dir_path):
        hr_height, hr_width = hr_shape

        self.tensor_setup = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )

        self.resize = transforms.Resize((512, 512), transforms.InterpolationMode.NEAREST)
        self.totensor = transforms.ToTensor()

        # get pass
        self.files_input = sorted(glob.glob(shadow_dir_path + "/*.*"))
        self.files_mask = sorted(glob.glob(mask_dir_path + "/*.*"))
        self.files_mask_edge = sorted(glob.glob(mask_edge_dir_path + "/*.*"))
        self.files_gt = sorted(glob.glob(gt_dir_path + "/*.*"))

        # check image numbers
        if abs(len(self.files_input) - len(self.files_mask)) + abs(len(self.files_input) - len(self.files_mask_edge)) + abs(len(self.files_input) - len(self.files_gt)) > 0:
            print("[ERROR] Image numbers are not the same. Please check path.")
            exit()

        if len(self.files_input) == 0 or len(self.files_mask) == 0 or len(self.files_mask_edge) == 0 or len(self.files_gt) == 0:
            print("[ERROR] We found no image. Please check path.")
            exit()


    def __getitem__(self, img_index):
        np.random.seed()

        rot = np.random.randint(0, 4)
        fliph = np.random.randint(0, 2)
        flipv = np.random.randint(0, 2)

        # Load images
        input_img = Image.open(self.files_input[img_index % len(self.files_input)])
        mask_img = Image.open(self.files_mask[img_index % len(self.files_mask)])
        mask_edge_img = Image.open(self.files_mask_edge[img_index % len(self.files_mask_edge)])
        gt_img = Image.open(self.files_gt[img_index % len(self.files_gt)])

        # crop
        if np.random.randint(0, 3) > 0:  # normal

            input_img = self.tensor_setup(input_img)
            mask_img = self.tensor_setup(mask_img)
            gt_img = self.tensor_setup(gt_img)
            mask_edge_img = self.tensor_setup(mask_edge_img)

        else:

            top = np.random.randint(0, 256)
            left = np.random.randint(0, 256)
            input_img = cropAndTotensor(input_img, top, left)
            mask_img = cropAndTotensor(mask_img, top, left)
            gt_img = cropAndTotensor(gt_img, top, left)
            mask_edge_img = cropAndTotensor(mask_edge_img, top, left)

        # flip
        input_img = rotflipImage(input_img, rot, flipv, fliph)
        mask_img = rotflipImage(mask_img, rot, flipv, fliph)
        gt_img = rotflipImage(gt_img, rot, flipv, fliph)
        mask_edge_img = rotflipImage(mask_edge_img, rot, flipv, fliph)

        # Color change
        if np.random.rand() > 0.4:
            # convert
            input_RGB = srgb2rgb(input_img)
            gt_RGB = srgb2rgb(gt_img)

            # calc diff
            diff = input_RGB / (gt_RGB + 1e-7)
            newOrder = np.array([0, 1, 2])
            np.random.shuffle(newOrder)
            new_gt_RGB = torch.zeros_like(gt_img)

            # change light strength
            rand_R = 0.6 + np.random.rand() * 0.8
            rand_G = 0.6 + np.random.rand() * 0.8
            rand_B = 0.6 + np.random.rand() * 0.8

            # color change mask
            c_mask = torch.ones_like(input_RGB)
            if np.random.rand() > 0.5:
                # change position
                c_mask_x = np.random.randint(0, 220)
                c_mask_y = np.random.randint(0, 220)
                c_mask_w = np.fmin(np.random.randint(60, 256) + c_mask_x, 256)
                c_mask_h = np.fmin(np.random.randint(60, 256) + c_mask_y, 256)
                # change light strength
                rand_R = 0.2 + np.random.rand() * 1.6
                rand_G = 0.2 + np.random.rand() * 1.6
                rand_B = 0.2 + np.random.rand() * 1.6
                c_mask[0, c_mask_x:c_mask_w, c_mask_y:c_mask_h] = rand_R
                c_mask[1, c_mask_x:c_mask_w, c_mask_y:c_mask_h] = rand_G
                c_mask[2, c_mask_x:c_mask_w, c_mask_y:c_mask_h] = rand_B

            # switch color layer
            new_gt_RGB[0, :, :] = gt_RGB[newOrder[0], :, :]
            new_gt_RGB[1, :, :] = gt_RGB[newOrder[1], :, :]
            new_gt_RGB[2, :, :] = gt_RGB[newOrder[2], :, :]
            new_gt_RGB *= c_mask

            # apply shadow
            new_input_RGB = new_gt_RGB * diff

            # clip
            new_input_RGB = torch.clamp(new_input_RGB, min=0.0, max=1.0)
            new_gt_RGB = torch.clamp(new_gt_RGB, min=0.0, max=1.0)

            # convert back
            input_img = rgb2srgb(new_input_RGB)
            gt_img = rgb2srgb(new_gt_RGB)

        return {"input_img": input_img, "gt_img": gt_img, "mask_img": mask_img, "mask_edge_img": mask_edge_img, "path": self.files_input[img_index % len(self.files_input)]}

    def __len__(self):

        return len(self.files_input)
