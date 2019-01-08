from models import ChainedPredictions, StackedHourGlass
import torch
import torchvision.transforms as transforms
from utils.data.images import ImageSequence
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

model_name = "resnet34"
hh_kernel = 1
oh_kernel = 1
n_joints = 16
n_channels = 256
n_stack = 2
n_modules = 2
n_reductions = 4
# pretrained_path = "../models/trainedModels/chained.pth"
pretrained_path = "../models/trainedModels/simpleHG.pth"
data_path = "../dataset/handtool_videos_minimal/barbell/barbell_0002/frames"

pretrained_path = os.path.abspath(os.path.join(os.curdir, pretrained_path))

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device_type)
device = torch.device(device_type)

# model = ChainedPredictions(model_name, hh_kernel, oh_kernel, n_joints)
model = StackedHourGlass(n_channels, n_stack, n_modules, n_reductions, n_joints)
model.to(device)
model.load_state_dict(torch.load(pretrained_path, map_location=device)['model_state'])
model.eval()

plt.show()

sequence = torch.utils.data.DataLoader(ImageSequence(data_path), batch_size=1, shuffle=False)

# frames = next(sequence)
# poses = model(frames)
# print('Generated')

for batch_img in sequence:
    batch_img = batch_img.to(device)
    # frame = torch.tensor(img).to(device)
    # frame = frame.unsqueeze(0)
    predicted_pose = model(batch_img).detach().cpu().numpy()
    print(predicted_pose.shape)
    # print(predicted_pose[0].shape)
    # print(predicted_pose[0][0, 0])
    # heatmaps = predicted_pose[0].detach().cpu().numpy()
    for k_heatmap in range(predicted_pose.shape[0]):
        heatmap = predicted_pose[k_heatmap]
        img = batch_img[k_heatmap].detach().cpu().numpy()
        joints = []
        plt.cla()
        plt.imshow(img.transpose((1, 2, 0)))
        for k in range(16):
            i, j = np.unravel_index(np.argmax(heatmap[k], axis=None), heatmap[k].shape)
            joints.append((i / heatmap[k].shape[0], j / heatmap[k].shape[1]))
            x, y = int(joints[-1][1] * img.shape[1]), int(joints[-1][0] * img.shape[2])
            circle = plt.Circle((x, y), radius=1, color='red')
            plt.gcf().gca().add_artist(circle)
        plt.draw()
        plt.pause(1)
    break
