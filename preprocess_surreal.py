import os
import h5py
from utils import Config
from utils.data import SurrealDataset, heatmat_to_2d_joints
from models import StackedHourGlass
import torch
import numpy as np
from tqdm import tqdm

config = Config('config/')

device_type = "cuda" if torch.cuda.is_available() and config.device_type == "cuda" else "cpu"

print("Using", device_type)

config_video_constraints = config.video_constraints
config_surreal = config.surreal

config = config.hourglass
device = torch.device(device_type)

path_to_data = "../dataset/surreal/data.hdf5"

model = StackedHourGlass(config.n_channels, config.n_stack, config.n_modules, config.n_reductions,
                         config.n_joints)
model.to(device)
model.load_state_dict(torch.load(config.pretrained_path, map_location=device)['model_state'])
model.eval()


def save_dataset(group, data_set):
    for k, (img, _, pose3d) in tqdm(enumerate(data_set), total=data_set.len):
        name = data_set.files[k].split('/')[-1].split('.')[0]
        poses = []
        for i in range(10, 50, config.batch_size):
            img_torch = torch.tensor(img[i:i + config.batch_size]).to(device)
            poses.append(heatmat_to_2d_joints(model(img_torch)[-1].detach().cpu().numpy()))
        predicted_pose = np.vstack(poses).transpose((1, 2, 0))
        group.create_dataset(name + "_2d", data=predicted_pose)
        group.create_dataset(name + "_3d", data=pose3d)


with h5py.File(path_to_data, "w") as f:
    # print("Save train")
    # train_group = f.create_group("train")
    # train_set = SurrealDataset(config_surreal.data_path, 'train', config_surreal.run)
    # save_dataset(train_group, train_set)
    print("Save test")
    test_group = f.create_group("test")
    test_set = SurrealDataset(config_surreal.data_path, 'test', config_surreal.run)
    test_set.len = 500
    save_dataset(test_group, test_set)
    # print("Save dev")
    # dev_group = f.create_group("dev")
    # dev_set = SurrealDataset(config_surreal.data_path, 'val', config_surreal.run)
    # dev_set.len = 50
    # save_dataset(dev_group, dev_set)
