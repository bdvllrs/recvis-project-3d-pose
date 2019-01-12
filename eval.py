from models import StackedHourGlass, Linear
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from utils.data import ImageSequence, SurrealDataset, get_order_joint_human, Human36M
from utils import Config, get_2d_joints, get_all_32joints
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import utils.viz as viz
from utils.eval import *

plt.ion()

config = Config('./config')

device_type = "cuda" if torch.cuda.is_available() and config.device_type == "cuda" else "cpu"

print("Using", device_type)

config_video_constraints = config.video_constraints
config_surreal = config.surreal

pretrained_path = os.path.abspath(os.path.join(os.curdir, config.hourglass.pretrained_path))
pretrained_path_linear = os.path.abspath(os.path.join(os.curdir, config.eval.linear_model))

device = torch.device(device_type)

# Stacked pre-trained model
hg_model = StackedHourGlass(config.hourglass.n_channels, config.hourglass.n_stack, config.hourglass.n_modules,
                            config.hourglass.n_reductions, config.hourglass.n_joints)
hg_model.to(device)
hg_model.load_state_dict(torch.load(pretrained_path, map_location=device)['model_state'])
hg_model.eval()

# 3D predictor
number_frames = 1
if config.eval.use_video_continuity:
    number_frames = config_video_constraints.frames_before + config_video_constraints.frames_after + 1
model = Linear(input_size=32 * number_frames, hidden_size=1024, output_size=48).to(device)
model.load_state_dict(torch.load(pretrained_path_linear, map_location=device))
model.to(device)

if config.eval.data.type == "sequence":
    sequence = torch.utils.data.DataLoader(ImageSequence(config.eval.data.path), batch_size=config.eval.batch_size,
                                           shuffle=False)
elif config.eval.data.type == "human":
    human_dataset = Human36M(config.eval.data.path, test_subjects=[config.eval.data.subject], actions=[config.eval.data.action])
    sequence = torch.utils.data.DataLoader(human_dataset.test_set, batch_size=config.eval.batch_size, shuffle=False)
else:
    raise ValueError(config.eval.data.type + " type does not exist.")

images = []
joints_2d = []
joints_3d = []
for batch in tqdm(sequence):  # size batch x 3 x 256 x 256
    if config.eval.data.type == "sequence":
        get_data_sequence(batch, device, hg_model, model, images, joints_2d, joints_3d)
    elif config.eval.data.type == "human":
        get_data_human(batch, device, human_dataset, model, images, joints_2d, joints_3d)


if len(images) > 0:
    images = np.vstack(images)
joints_2d = np.vstack(joints_2d)  # shape batch x 32 * 2
joints_3d = np.vstack(joints_3d)  # shape batch x 32 * 3

"""
0 LFOOT, 1 RHIP, 2 RKNEE, 3 RFOOT, 4 -1, 5 -1, 6 0, 7 LHIP, 8 LKNEE, 9 -1, 10 -1, 11 -1, 12 HIP, 13 Thorax, 14 -1, 15 Neck, 16 -1, 17 RSHoulder
18 LShoulder, 19 LELBOX, 20 -1, 21 -1, 22 -1, 23 -1, 24 -1, 25 RELBOW, 26 RHAND, 27 TOP HEAD, 28 -1, 29 -1, 30 -1, 31 -1
"""

for t in range(5, joints_2d.shape[0]):
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
    plt.axis('off')

    if len(images) > 0:
        ax1 = plt.subplot(gs1[0])
        ax1.imshow(images[t].transpose((1, 2, 0)))

    ax2 = plt.subplot(gs1[1])
    viz.show2Dpose(joints_2d[t], ax2)

    ax3 = plt.subplot(gs1[2], projection='3d')
    viz.show3Dpose(joints_3d[t], ax3, lcolor="#9b59b6", rcolor="#2ecc71")

    plt.draw()
    plt.pause(20)
