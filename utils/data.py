"""
Adapted from @una-dinosauria (Julieta Martinez) data_utils.py file
"""
import copy
import re
import os
import h5py
import numpy as np
from utils import project_to_cameras, transform_world_to_camera, load_camera_params
import torch.utils.data
import torch
import matplotlib.pyplot as plt
import utils.viz as viz

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = [''] * 16
SH_NAMES[0] = 'RFoot'
SH_NAMES[1] = 'RKnee'
SH_NAMES[2] = 'RHip'
SH_NAMES[3] = 'LHip'
SH_NAMES[4] = 'LKnee'
SH_NAMES[5] = 'LFoot'
SH_NAMES[6] = 'Hip'
SH_NAMES[7] = 'Spine'
SH_NAMES[8] = 'Thorax'
SH_NAMES[9] = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

ACTIONS = ["Directions", "Discussion", "Eating", "Greeting",
           "Phoning", "Photo", "Posing", "Purchases",
           "Sitting", "SittingDown", "Smoking", "Waiting",
           "WalkDog", "Walking", "WalkTogether"]


def normalize_stats(poses, dim, predict_14=False):
    """
     Computes normalization statistics: mean and stdev, dimensions used and ignored
     Args:
       poses: nxd np array with poses
       dim: 2 or 3
       predict_14: boolean. Whether to use only 14 joints
     Returns:
       data_mean: np vector with the mean of the data
       data_std: np vector with the standard deviation of the data
       dimensions_to_ignore: list of dimensions not used in the model
       dimensions_to_use: list of dimensions used in the model
     """
    data_mean = np.mean(poses, axis=0)
    data_std = np.std(poses, axis=0)

    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    if dim == 2:
        dimensions_to_use = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2), dimensions_to_use)
    else:  # 3
        dimensions_to_use = np.where(np.array([x != '' for x in H36M_NAMES]))[0]
        dimensions_to_use = np.delete(dimensions_to_use, [0, 7, 9] if predict_14 else 0)

        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 3,
                                               dimensions_to_use * 3 + 1,
                                               dimensions_to_use * 3 + 2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3), dimensions_to_use)

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def normalize_data(data, data_mean, data_std, dim_to_use):
    """
    Normalizes a dictionary of poses
    Args:
        data: dictionary where values are
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions to keep in the data
    Returns:
        data_out: dictionary with same keys as data, but values have been normalized
    """
    data_out = {}

    for key in data.keys():
        data[key] = data[key][:, dim_to_use]
        mu = data_mean[dim_to_use]
        stddev = data_std[dim_to_use]
        data_out[key] = np.divide((data[key] - mu), stddev)
    return data_out


def un_normalize_data(normalized_data, data_mean, data_std, dimensions_to_ignore):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation. Some dimensions might also be missing
    Args:
        normalized_data: nxd matrix to unnormalize
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dimensions_to_ignore: list of dimensions that were removed from the original data
    Returns:
        orig_data: the input normalized_data, but unnormalized
    """
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # Dimensionality

    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                  if dim not in dimensions_to_ignore])

    orig_data[:, dimensions_to_use] = normalized_data

    # Multiply times stdev and add the mean
    std_mat = data_std.reshape((1, D))
    std_mat = np.repeat(std_mat, T, axis=0)
    mean_mat = data_mean.reshape((1, D))
    mean_mat = np.repeat(mean_mat, T, axis=0)
    orig_data = np.multiply(orig_data, std_mat) + mean_mat
    return orig_data


def postprocess_3d(poses_set):
    """
    Center 3d points around root
    Args:
        poses_set: dictionary with 3d data
    Returns:
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:, :3])

        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile(poses[:, :3], [1, len(H36M_NAMES)])
        poses_set[k] = poses

    return poses_set, root_positions


def get_array(poses_2d, poses_3d, root_positions, camera_frame, for_video=False, frames_before=0, frames_after=0):
    """
    Get array from poses dict
    Args:
        poses_2d:
        poses_3d:
        root_positions:
        camera_frame:
        for_video: if True, uses video continuity constraints
        frames_before: if for_video, number of frames to use before current one
        frames_after: if for_video, number of frames to use after current one
    """
    stack_poses, stack_targets, stack_root_positions = [], [], []
    keys = []
    action_to_keys = {}
    for key in sorted(poses_2d.keys()):
        subj, action, seqname = key
        key3d = key if camera_frame else (subj, action, '{0}.h5'.format(seqname.split('.')[0]))
        for k in range(frames_before, poses_2d[key].shape[0] - frames_after - 1 * int(for_video)):
            input_poses = poses_2d[key][k]
            if for_video:
                input_poses = []
                for s in range(2):
                    in_2d_poses = [poses_2d[key][k + s]]
                    for i in range(1, frames_before + 1):
                        in_2d_poses.append(poses_2d[key][k + s - i])
                    for i in range(1, frames_after + 1):
                        in_2d_poses.append(poses_2d[key][k + s + i])
                    input_poses.append(in_2d_poses)
            stack_poses.append(input_poses)
            stack_targets.append(poses_3d[key3d][k])
            stack_root_positions.append(root_positions[key][k])
            if action not in action_to_keys.keys():
                action_to_keys[action] = []
            action_to_keys[action].append(len(stack_poses) - 1)
            keys.append(key)
    stack_poses = np.stack(stack_poses, axis=0)
    stack_targets = np.stack(stack_targets, axis=0)
    stack_root_positions = np.stack(stack_root_positions, axis=0)
    return (stack_poses,
            stack_targets,
            stack_root_positions,
            action_to_keys, keys)


class Human36M:
    def __init__(self, path, train_subjects=None, test_subjects=None, actions=None, use_camera_frame=True,
                 max_video_length=-1, use_hourglass=False, video_constraints=False, frames_before=0, frames_after=0):

        self.train_subjects = train_subjects if train_subjects is not None else TRAIN_SUBJECTS
        self.test_subjects = test_subjects if test_subjects is not None else TEST_SUBJECTS
        self.actions = actions if actions is not None else ACTIONS

        self.max_video_length = max_video_length
        self.video_constraints = video_constraints
        self.frames_before = frames_before
        self.frames_after = frames_after

        self.file_path = os.path.abspath(os.path.join(os.curdir, path))
        self.camera_path = os.path.abspath(os.path.join(os.curdir, path, 'cameras.h5'))

        print("Loading data...")
        self.train_poses = self.load_joints(self.train_subjects, self.actions)
        self.test_poses = self.load_joints(self.test_subjects, self.actions)
        self.cameras = self.load_cameras()
        print("Loading 2D...")
        input_train, input_test, self.data_mean_2d, self.data_std_2d, self.dim_to_ignore_2d, self.dim_to_use_2d = self.get_2d(
            use_hourglass)
        print("Loading 3D...")
        (output_train, output_test, self.data_mean_3d, self.data_std_3d, self.dim_to_ignore_3d, self.dim_to_use_3d,
         train_root_positions, test_root_positions) = self.get_3d(camera_frame=use_camera_frame)
        self.train_set = Dataset(input_train, output_train, train_root_positions, use_camera_frame=use_camera_frame,
                                 video_constraints=video_constraints, frames_before=frames_before,
                                 frames_after=frames_after)
        self.test_set = Dataset(input_test, output_test, test_root_positions,
                                use_camera_frame=use_camera_frame, video_constraints=video_constraints,
                                frames_before=frames_before, frames_after=frames_after)
        print("Loaded.")

    def load_joints(self, subjects=None, actions=None):
        """
        Loads the 3D joints of the dataset.
        Sets self.poses as a dictionary of keys (subject, action, file name) and values of size (N, 96 = 32x3).
        There are 32 joints of (x, y, z) values.
        Args:
            subjects: subjects to consider
            actions: actions to load
        """
        poses = {}
        total_len = 0
        for subject in subjects:
            for action in actions:
                path = os.path.join(self.file_path,
                                    'S{0}'.format(subject),
                                    'MyPoses/3D_positions')
                request = r'{0}\s?[0-9]?.h5'.format(action)
                file_names = [f for f in os.listdir(path) if re.search(request, f)]
                for filename in file_names:
                    with h5py.File(os.path.join(path, filename), 'r') as file:
                        total_len += file['3D_positions'].shape[1]
                        poses[(subject, action, filename)] = file['3D_positions'][:, :self.max_video_length].T
        return poses

    def load_hourglass(self, subjects, actions):
        """
        Load 2D poses from the hourglass
        Args:
            subjects:
            actions:
        """
        hourglass_to_human_joints = np.array([SH_NAMES.index(h) for h in H36M_NAMES if h != '' and h in SH_NAMES])
        assert np.all(hourglass_to_human_joints == np.array([6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10]))
        data = {}
        for subject in subjects:
            for action in actions:
                path = os.path.join(self.file_path,
                                    'S{0}'.format(subject),
                                    'StackedHourglass')
                request = r'{0}_?[0-9]?.[0-9]+.h5'.format(action)
                file_names = [f for f in os.listdir(path) if re.search(request, f)]
                for filename in file_names:
                    seqname = filename.replace('_', ' ')
                    with h5py.File(os.path.join(path, filename), 'r') as file:
                        poses = file['poses'][:]

                        # Permute the loaded data to make it compatible with H36M
                        poses = poses[:, hourglass_to_human_joints, :]

                        # Reshape into n x (32*2) matrix
                        poses = np.reshape(poses, [poses.shape[0], -1])
                        poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 2])

                        dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
                        dim_to_use_y = dim_to_use_x + 1

                        dim_to_use = np.zeros(len(SH_NAMES) * 2, dtype=np.int32)
                        dim_to_use[0::2] = dim_to_use_x
                        dim_to_use[1::2] = dim_to_use_y
                        poses_final[:, dim_to_use] = poses
                        seqname = seqname + '-sh'
                        data[(subject, action, seqname)] = poses_final
        return data

    def load_cameras(self):
        """Loads the cameras of h36m
        Args
          bpath: path to hdf5 file with h36m camera data
          subjects: List of ints representing the subject IDs for which cameras are requested
        Returns
          rcams: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
        """
        rcams = {}
        subjects = self.train_subjects[:]
        subjects.extend(self.test_subjects[:])

        with h5py.File(self.camera_path, 'r') as cameras:
            for subject in subjects:
                for camera in range(4):  # There are 4 cameras in human3.6m
                    rcams[(subject, camera + 1)] = load_camera_params(cameras,
                                                                      'subject%d/camera%d/{0}' % (subject, camera + 1))
        return rcams

    def get_2d(self, use_hourglass=False):
        """
        Creates 2d poses by projecting 3d poses with the corresponding camera
        parameters. Also normalizes the 2d poses
        Returns:
            train_set: dictionary with projected 2d poses for training
            test_set: dictionary with projected 2d poses for testing
            data_mean: vector with the mean of the 2d training data
            data_std: vector with the standard deviation of the 2d training data
            dim_to_ignore: list with the dimensions to not predict
            dim_to_use: list with the dimensions to predict
        """

        # Load 3d data
        if not use_hourglass:
            train_set = project_to_cameras(self.train_poses, self.cameras, H36M_NAMES)
            test_set = project_to_cameras(self.test_poses, self.cameras, H36M_NAMES)
        else:
            train_set = self.load_hourglass(self.train_subjects, self.actions)
            test_set = self.load_hourglass(self.test_subjects, self.actions)

        # Compute normalization statistics.
        complete_train = copy.deepcopy(np.vstack(train_set.values()))
        data_mean, data_std, dim_to_ignore, dim_to_use = normalize_stats(complete_train, dim=2)

        # Divide every dimension independently
        train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
        test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

        return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use

    def get_3d(self, camera_frame, predict_14=False):
        """
        Loads 3d poses, zero-centres and normalizes them
        Args:
            camera_frame: boolean. Whether to convert the data to camera coordinates
            predict_14: boolean. Whether to predict only 14 joints
        Returns:
            train_set: dictionary with loaded 3d poses for training
            test_set: dictionary with loaded 3d poses for testing
            data_mean: vector with the mean of the 3d training data
            data_std: vector with the standard deviation of the 3d training data
            dim_to_ignore: list with the dimensions to not predict
            dim_to_use: list with the dimensions to predict
            train_root_positions: dictionary with the 3d positions of the root in train
            test_root_positions: dictionary with the 3d positions of the root in test
        """
        if camera_frame:
            train_set = transform_world_to_camera(self.train_poses, self.cameras, H36M_NAMES)
            test_set = transform_world_to_camera(self.test_poses, self.cameras, H36M_NAMES)
        else:
            train_set = self.train_poses
            test_set = self.test_poses

        # Apply 3d post-processing (centering around root)
        train_set, train_root_positions = postprocess_3d(train_set)
        test_set, test_root_positions = postprocess_3d(test_set)

        # Compute normalization statistics
        complete_train = copy.deepcopy(np.vstack(train_set.values()))
        data_mean, data_std, dim_to_ignore, dim_to_use = normalize_stats(complete_train, dim=3,
                                                                         predict_14=predict_14)

        # Divide every dimension independently
        train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
        test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

        return (train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use,
                train_root_positions, test_root_positions)


class Dataset:
    def __init__(self, data, targets, root_positions, use_camera_frame=True, video_constraints=False, frames_before=0,
                 frames_after=0):
        (self.data, self.targets,
         self.root_positions,
         self.action_to_keys, self.keys) = get_array(data, targets, root_positions, use_camera_frame,
                                                     video_constraints, frames_before, frames_after)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return (torch.tensor(self.data[item]),
                torch.tensor(self.targets[item]),
                torch.tensor(self.root_positions[item]),
                self.keys[item])
