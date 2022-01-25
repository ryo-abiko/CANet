'''
Test code of CANet
'''

import glob
import os
from PIL import Image
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

from network import network


class testImageDataset(Dataset):
    def __init__(self, path, resized):
        
        self.resized = resized

        self.tensor_setup_resized = transforms.Compose(
            [
                transforms.Resize((256, 256), transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )

        # get image path
        self.files_input = sorted(glob.glob(path + "/*.*"))

        # If we can use GPU
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.backends.cudnn.benchmark = True
        else:
            self.device = "cpu"

    def __getitem__(self, img_index):

        # Load Images
        input_img = Image.open(self.files_input[img_index % len(self.files_input)])
        if self.resized == 0:
            input_img = tf.to_tensor(input_img)
        else:
            input_img = self.tensor_setup_resized(input_img)

        # Get image path
        imgPath = os.path.splitext(os.path.basename(self.files_input[img_index % len(self.files_input)]))[0]

        return {"input_img": input_img.to(self.device), "path": imgPath}

    def __len__(self):

        return len(self.files_input)

# ------- settings --------
parser = argparse.ArgumentParser()
parser.add_argument("--detection_pth_path", type=str, default="", help="Folder name to evaluate")
parser.add_argument("--removal_pth_path", type=str, default="", help="Folder Name of the dataset")
parser.add_argument("--input_img_path", type=str, default="", help="Folder Name of the dataset")
parser.add_argument("--resized", type=int, default=1, help="0: not resized, 1: resized to 256x256")
opt = parser.parse_args()
# -------------------------

# Create output directory
os.makedirs("output_CANet", exist_ok=True)

# init Network for test
myNetwork = network()
myNetwork.initForTest()

# load model
if not myNetwork.loadModel(d_path=opt.detection_pth_path, r_path=opt.removal_pth_path):
    print("[ERROR] No weight is found.")
    exit()

# prepare dataset
image_dataset = testImageDataset(opt.input_img_path, opt.resized)
if len(image_dataset) == 0:
    print("[ERROR] No test image is found.")
    exit()

# test image
for image_num in range(len(image_dataset)):

    imgs = image_dataset[image_num]

    output = myNetwork.test(imgs["input_img"])

    print(image_num)

    # Save Image data
    output_np = output.to("cpu").detach().numpy().copy()
    output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
    output_pil.save("output_CANet/" + imgs["path"] + ".png")
