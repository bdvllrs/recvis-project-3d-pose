import torch
import numpy as np

from utils import get_2d_joints, get_all_32joints
from utils.data import un_normalize_data, H36M_NAMES

dim_to_use_2d = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 30, 31, 34, 35, 36, 37,
                 38, 39, 50, 51, 52, 53, 54, 55]
dim_to_use_3d = [3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 36, 37, 38, 39, 40, 41,
                 42, 43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 75, 76, 77, 78, 79, 80, 81, 82, 83]

dim_to_ignore_2d = np.delete(np.arange(len(H36M_NAMES) * 2), dim_to_use_2d)
dim_to_ignore_3d = np.delete(np.arange(len(H36M_NAMES) * 3), dim_to_use_3d)

data_mean_3d = np.load('models/mean_3d.npy')
data_std_3d = np.load('models/std_3d.npy')


def get_data_sequence(batch, device, hg_model, model, images, joints_2d, joints_3d, conf):
    batch = batch.to(device)
    images.append(batch.detach().cpu().numpy())
    predicted_2d_poses = get_all_32joints(get_2d_joints(hg_model, batch), 2, dim_to_ignore_2d)  # batch x 16 x 2
    joints_2d.append(predicted_2d_poses)
    # Normalize
    # data_mean = np.mean(predicted_2d_poses, axis=0)
    # data_std = np.std(predicted_2d_poses, axis=0)
    predicted_2d_poses = predicted_2d_poses[:, dim_to_use_2d]
    # mu = data_mean[dim_to_use_2d]
    # stddev = data_std[dim_to_use_2d]
    # predicted_2d_poses = np.divide((predicted_2d_poses - mu), stddev)

    # Apply our model
    poses_2d = torch.tensor(predicted_2d_poses).to(device, torch.float)
    poses_3d = model(poses_2d).detach().cpu().numpy()
    # poses_3d = get_all_32joints(poses_3d, 3, dim_to_ignore_3d)
    poses_3d = un_normalize_data(poses_3d, data_mean_3d,
                                 data_std_3d, dim_to_ignore_3d)
    # poses_3d = poses_3d.reshape(poses_3d.shape[0], 16, 3)
    joints_3d.append(poses_3d)


def get_data_human(batch, device, human_dataset, model, images, joints_2d, joints_3d, conf):
    data_2d, data_3d, root_position, keys = batch
    data_2d_cur = data_2d[:, 0, conf.eval.video_constraints.frames_before]
    data_2d = data_2d[:, 0].reshape(data_2d.size(0), -1)
    predicted_3d = model(data_2d.to(device, torch.float)).detach().cpu().numpy()
    data_2d_un = un_normalize_data(data_2d_cur, human_dataset.data_mean_2d,
                                   human_dataset.data_std_2d, human_dataset.dim_to_ignore_2d)
    data_3d_pred_un = un_normalize_data(predicted_3d, human_dataset.data_mean_3d,
                                        human_dataset.data_std_3d, human_dataset.dim_to_ignore_3d)
    joints_2d.append(data_2d_un)
    joints_3d.append(data_3d_pred_un)
