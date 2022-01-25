'''
Main file of CANet
'''
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms.functional as tf

import models
import getDataset


def gridImage(imgs):

    img_grid = make_grid(imgs[0], nrow=1, normalize=False)

    for i in range(1, len(imgs)):
        img_grid = torch.cat((img_grid, make_grid(imgs[i], nrow=1, normalize=False)), -1)

    return img_grid


def convert_to_numpy(input_temp_img, H=-1, W=-1):

    if H == -1:
        H = input_temp_img.shape[-2]
    if W == -1:
        W = input_temp_img.shape[-1]

    if len(input_temp_img.shape) == 4:
        colorCNL = input_temp_img.shape[1]
        return input_temp_img[:, :, :H, :W].detach().clone().cpu().numpy().reshape(colorCNL, H, W).transpose(1, 2, 0)

    colorCNL = input_temp_img.shape[0]
    return input_temp_img[:, :H, :W].detach().clone().cpu().numpy().reshape(colorCNL, H, W).transpose(1, 2, 0)


def concat_np_images(imgs):

    outimage = imgs[0]
    if len(imgs) > 1:
        for i in range(1, len(imgs)):
            if imgs[i].shape[2] == 1:
                imgs[i].reshape(imgs[i].shape[0], imgs[i].shape[1], 1)
                outimage = np.concatenate([outimage, np.concatenate([imgs[i], imgs[i], imgs[i]], 2)], 1)
            else:
                outimage = np.concatenate([outimage, imgs[i]], 1)

    return outimage


class network():
    def __init__(self, train_name="test"):
        self.train_name = train_name

        # GPU or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
        else:
            self.device = "cpu"

        # load model
        self.generator_d = models.GeneratorUNet_d().to(self.device)
        self.generator_r = models.GeneratorUNet_r().to(self.device)

    def initForTrain(self, train_batchsize, shadow_dir_path, gt_dir_path, mask_dir_path, mask_edge_dir_path):

        # Create directories
        os.makedirs("train_output", exist_ok=True)
        os.makedirs("train_output/" + self.train_name, exist_ok=True)
        os.makedirs("train_output/" + self.train_name + "/images", exist_ok=True)
        os.makedirs("train_output/" + self.train_name + "/saved_models", exist_ok=True)

        # load discriminator model
        hr_shape = (256, 256)
        self.discriminator_d = models.Discriminator_d(input_shape=(4, *hr_shape)).to(self.device)
        self.discriminator_r = models.Discriminator_r(input_shape=(9, *hr_shape)).to(self.device)

        # create loss model
        self.criterion_GAN = torch.nn.MSELoss().to(self.device)
        self.criterion_mse = torch.nn.MSELoss().to(self.device)

        # create optimizer
        self.optimizer_G_d = torch.optim.Adam(self.generator_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_G_r = torch.optim.Adam(self.generator_r.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_d = torch.optim.Adam(self.discriminator_d.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D_r = torch.optim.Adam(self.discriminator_r.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # load dataset for train
        self.dataloader = DataLoader(
            getDataset.ImageDataset(hr_shape, shadow_dir_path, gt_dir_path, mask_dir_path, mask_edge_dir_path),
            batch_size=train_batchsize,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def initForTest(self):
        # Set generator to eval mode
        self.generator_d.eval()
        self.generator_r.eval()

    def loadModel(self, d_path, r_path):

        existFlg = True
        # detection
        if os.path.exists(d_path):
            self.generator_d.load_state_dict(torch.load(d_path), strict=False)
            print("Loaded" + d_path + "for detection net.")
        else:   
            existFlg = False

        # removal
        if os.path.exists(r_path):
            self.generator_r.load_state_dict(torch.load(r_path), strict=False)
            print("Loaded" + r_path + "for removal net.")
        else:
            existFlg = False

        return existFlg

    def train(self, images, batches_done):
        # init loss
        self.loss = {}

        # get train image
        self.input_img = images["input_img"].to(self.device)
        self.mask_img = images["mask_img"].to(self.device)
        self.gt_img = images["gt_img"].to(self.device)
        self.mask_edge_img = images["mask_edge_img"].to(self.device)

        valid = torch.randn((self.input_img.size(0), *self.discriminator_d.output_shape), device=self.device, requires_grad=False) * 0.01 + 0.5  # real
        fake = torch.randn((self.input_img.size(0), *self.discriminator_d.output_shape), device=self.device, requires_grad=False) * 0.01 - 0.5  # fake

        # ------------------------
        #  Generater detection net
        # ------------------------
        self.optimizer_G_d.zero_grad()
        self.optimizer_G_r.zero_grad()
        self.generator_r.eval()

        self.generated_mask = self.generator_d(self.input_img)
        loss_G_mse = self.criterion_mse(self.generated_mask, self.mask_img)

        loss_G_adv = self.criterion_GAN(
            self.discriminator_d(torch.cat([self.input_img, self.generated_mask], 1)),
            valid
        )

        # loss_G_feat
        loss_G_feat = self.criterion_mse(
            self.generator_r(torch.cat([self.input_img, self.input_img * self.generated_mask], 1)),
            self.gt_img
        )

        loss_G1 = (loss_G_mse + loss_G_adv * 10 / (batches_done + 1) + 10 * loss_G_feat)
        loss_G1.backward()
        self.optimizer_G_d.step()

        self.loss["loss_G1"] = loss_G1.detach().item()

        # ------------------------
        #  Generater removal net
        # ------------------------
        self.optimizer_G_r.zero_grad()
        self.generator_r.train()

        self.output = self.generator_r(torch.cat([self.input_img, self.input_img * self.generated_mask.detach()], 1))

        loss_G_r_mse = self.criterion_mse(
            self.output,
            self.gt_img
        )

        loss_G_r_edge = self.criterion_mse(
            self.output * self.mask_edge_img.detach(),
            self.gt_img * self.mask_edge_img.detach()
        )

        loss_G_r_adv = self.criterion_GAN(
            self.discriminator_r(torch.cat([self.input_img, self.input_img * self.generated_mask.detach(), self.output], 1)),
            valid
        )

        loss_G2 = loss_G_r_mse + 25 * loss_G_r_edge + loss_G_r_adv / 10 / np.sqrt(batches_done + 1)
        loss_G2.backward()
        self.optimizer_G_r.step()

        self.loss["loss_G2"] = loss_G2.detach().item()
        self.loss["loss_G_r_mse"] = loss_G_r_mse.detach().item()
        self.loss["loss_G_r_edge"] = 25 * loss_G_r_edge.detach().item()
        self.loss["loss_G_r_adv"] = loss_G_r_adv.detach().item() / 10 / np.sqrt(batches_done + 1)

        # ---------------------
        # Discriminator detection net
        # ---------------------
        self.optimizer_D_d.zero_grad()

        generated_mask = self.generator_d(self.input_img).detach()

        loss_real = self.criterion_GAN(self.discriminator_d(torch.cat([self.input_img, self.mask_img], 1)), valid)
        loss_fake = self.criterion_GAN(self.discriminator_d(torch.cat([self.input_img, generated_mask], 1)), fake)

        # Total loss
        loss_D_d = (loss_real + loss_fake) / 2

        loss_D_d.backward()
        self.optimizer_D_d.step()

        self.loss["loss_D_d"] = loss_D_d.detach().item()

        # ---------------------
        # Discriminator removal net
        # ---------------------
        self.optimizer_D_r.zero_grad()

        input_G_r = torch.cat([self.input_img, self.input_img * generated_mask], 1)
        output = self.generator_r(input_G_r).detach()

        loss_real = self.criterion_GAN(self.discriminator_r(torch.cat([self.input_img, self.input_img * self.mask_img, self.gt_img], 1)), valid)
        loss_fake = self.criterion_GAN(self.discriminator_r(torch.cat([self.input_img, self.input_img * generated_mask, output], 1)), fake)

        # Total loss
        loss_D_r = (loss_real + loss_fake) / 2

        loss_D_r.backward()
        self.optimizer_D_r.step()

        self.loss["loss_D_r"] = loss_D_r.detach().item()

    def test(self, input):

        input_img = torch.unsqueeze(input, 0).to(self.device)

        # select output mode
        input_imgs = input_img

        angles = [90, 180, 270]
        for angle in angles:
            input_imgs = torch.cat([input_imgs, tf.rotate(input_img, angle)], 0)
        
        with torch.no_grad():
            mask_imgs = self.generator_d(input_imgs).detach()
            output_imgs = self.generator_r(torch.cat([input_imgs, input_imgs * mask_imgs], 1))

        for i, angle in enumerate(angles):
            output_imgs[i + 1, :, :, :] = tf.rotate(output_imgs[i + 1, :, :, :], -angle)

        output_imgs = torch.clamp(output_imgs, min=0.0, max=1.0)
        self.output = torch.mean(output_imgs, 0).permute(1, 2, 0)

        return self.output


    def getResultGrid(self, imgs):

        input_img = imgs["input_img"]
        gt_img = imgs["gt_img"]

        output = self.test(input_img)

        # convert to numpy
        input_img = convert_to_numpy(input_img)
        gt_img = convert_to_numpy(gt_img)
        mask = convert_to_numpy(self.mask)

        # concat images
        img_grid = concat_np_images([input_img, mask, output, gt_img, np.abs(output - gt_img)])
        return img_grid

    def logProcess(self, epoch, batch):
        sumLoss = self.loss["loss_G_r_mse"] + self.loss["loss_G_r_edge"] + self.loss["loss_G_r_adv"]
        print(
            "[%s] [Epoch %d] [Batch %d/%d] [MSE loss: %.2f]  [Edge loss: %.2f] [Adv loss: %.2f] [sum: %f]"
            % (self.train_name, epoch, batch, len(self.dataloader), self.loss["loss_G_r_mse"] / sumLoss * 100, self.loss["loss_G_r_edge"] / sumLoss * 100, self.loss["loss_G_r_adv"] / sumLoss * 100, sumLoss)
        )

    def saveImage(self, batches_done):
        img_grid = gridImage((self.input_img.detach(), self.generated_mask.detach(), self.output.detach(), self.gt_img.detach()))
        save_image(img_grid, "train_output/" + self.train_name + "/images/" + str(batches_done).zfill(10) + ".jpg", normalize=False)

    def saveWeight(self, epoch):
        # Save model checkpoints
        os.makedirs("train_output/{}/saved_models/{}".format(self.train_name, str(epoch)), exist_ok=True)
        torch.save(self.generator_d.state_dict(), "train_output/{}/saved_models/{}/generator_d_{}.pth".format(self.train_name, str(epoch), str(epoch)))
        torch.save(self.generator_r.state_dict(), "train_output/{}/saved_models/{}/generator_r_{}.pth".format(self.train_name, str(epoch), str(epoch)))
