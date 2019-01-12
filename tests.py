from models import ChainedPredictions, StackedHourGlass
import torch
import torchvision.transforms as transforms
from utils.data import ImageSequence, SurrealDataset, get_order_joint_human
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import utils.viz as viz

# plt.ion()

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
data_path = "../dataset/handtool_videos_minimal/barbell/barbell_0010/frames"
data_path_surreal = "/run/media/bdvllrs/Data/Documents/Supelec/MVA/Image/SURREAL/data"

pretrained_path = os.path.abspath(os.path.join(os.curdir, pretrained_path))

device_type = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device_type)
device = torch.device(device_type)

# model = ChainedPredictions(model_name, hh_kernel, oh_kernel, n_joints)
model = StackedHourGlass(n_channels, n_stack, n_modules, n_reductions, n_joints)
model.to(device)
model.load_state_dict(torch.load(pretrained_path, map_location=device)['model_state'])
model.eval()

# plt.show()

# sequence = torch.utils.data.DataLoader(ImageSequence(data_path), batch_size=1, shuffle=False)
sequence = torch.utils.data.DataLoader(SurrealDataset(data_path_surreal, "val", "run0"), batch_size=1, shuffle=False)

# frames = next(sequence)
# poses = model(frames)
# print('Generated')
# for img, _, points_3d in sequence:
#     points_3d_img1 = points_3d[0, :, :, 0].detach().cpu().numpy()
#     img = img[0, 0].detach().cpu().numpy().transpose((1, 2, 0))
#     points_3d_img1 = get_order_joint_human(points_3d_img1).transpose((1, 0))
#     points_3d_img1 = points_3d_img1.reshape(-1)
#     # print(points_3d_img1.shape)
#     # Visualize random samples
#     # 1080p	= 1,920 x 1,080
#     fig = plt.figure(1)
#
#     gs1 = gridspec.GridSpec(1, 2)  # 5 rows, 9 columns
#     gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
#     plt.axis('off')
#     ax2 = plt.subplot(gs1[0])
#     ax2.imshow(img)
#
#     ax3 = plt.subplot(gs1[1], projection='3d')
#     viz.show3Dpose(points_3d_img1, ax3, lcolor="#9b59b6", rcolor="#2ecc71")
#
#     plt.draw()
#     plt.pause(5)

for batch_img, _, _ in sequence:
    batch_img = batch_img.to(device)
    # frame = torch.tensor(img).to(device)
    # frame = frame.unsqueeze(0)
    for l in range(batch_img.shape[1]):
        predicted_pose = model(batch_img[:, l])[-1].detach().cpu().numpy()
        # print(predicted_pose[0].shape)
        # print(predicted_pose[0][0, 0])
        # heatmaps = predicted_pose[0].detach().cpu().numpy()
        for k_heatmap in range(predicted_pose.shape[0]):
            heatmap = predicted_pose[k_heatmap]
            img = batch_img[0, l].detach().cpu().numpy()
            joints = []
            for k in range(16):
                plt.cla()
                plt.imshow(img.transpose((1, 2, 0)))
                i, j = np.unravel_index(np.argmax(heatmap[k], axis=None), heatmap[k].shape)
                joints.append((i / heatmap[k].shape[0], j / heatmap[k].shape[1]))
                x, y = int(joints[-1][1] * img.shape[1]), int(joints[-1][0] * img.shape[2])
                circle = plt.Circle((x, y), radius=1, color='red')
                plt.gcf().gca().add_artist(circle)
                plt.show()
            plt.draw()
            plt.pause(0.01)
    break
