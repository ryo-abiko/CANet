'''
Train code of CANet
'''

import argparse
import os

from network import network

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--train_name", type=str, default="test_train", help="training name")
PARSER.add_argument("--shadow_dir_path", type=str, default="", help="path to training shadow images")
PARSER.add_argument("--gt_dir_path", type=str, default="", help="path to training groundtruth images")
PARSER.add_argument("--mask_dir_path", type=str, default="", help="path to training mask images")
PARSER.add_argument("--mask_edge_dir_path", type=str, default="", help="path to training mask edge images")
PARSER.add_argument("--batch_size", type=int, default=4, help="size of the batches")
PARSER.add_argument("--sample_interval", type=int, default=1000, help="interval between saving image samples")
PARSER.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
PARSER.add_argument("--pretrained_d", type=str, default="", help="pretrained detection net")
PARSER.add_argument("--pretrained_r", type=str, default="", help="pretrained removal net")
OPT = PARSER.parse_args()
print(OPT)

myNetwork = network(OPT.train_name)
myNetwork.initForTrain(OPT.batch_size, OPT.shadow_dir_path, OPT.gt_dir_path, OPT.mask_dir_path, OPT.mask_edge_dir_path)

# Load pretrained models
myNetwork.loadModel(OPT.pretrained_d, OPT.pretrained_r)

# ----------
#  Training
# ----------
for epoch in range(0, 100000):
    for i, imgs in enumerate(myNetwork.dataloader):

        batches_done = epoch * len(myNetwork.dataloader) + i

        # Train model
        myNetwork.train(imgs, batches_done)

        # Log Progress
        if batches_done % 10 == 0:
            myNetwork.logProcess(epoch, i)

        # save trained image
        if batches_done % OPT.sample_interval == 0:
            myNetwork.saveImage(batches_done)

    # Save weight
    if OPT.checkpoint_interval != -1 and epoch % OPT.checkpoint_interval == 0:
        myNetwork.saveWeight(epoch)
