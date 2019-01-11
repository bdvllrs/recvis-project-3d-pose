import numpy as np
import cv2
import os
from scipy.io import loadmat


def get_frames_from_video(video_path, scale):
    """
    Get frame generator from video file path
    Args:
        video_path: path of the video
        scale: scale the image

    Returns: generator of the frames
    """
    video = cv2.VideoCapture(video_path)
    has_next_frame, frame = video.read()
    while has_next_frame:
        # Rescale the image
        frame = cv2.resize(frame, dsize=(int(320 * scale), int(240 * scale)), interpolation=cv2.INTER_CUBIC)
        yield frame.astype(np.float32)
        has_next_frame, frame = video.read()


def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def center_crop(img, joints_2d, size_x, size_y):
    """

    Args:
        joints_2d:
        img: shape (T, width, height, 3)
        size_x:
        size_y:
    """
    start_x = img.shape[2] // 2 - (size_x // 2)
    start_y = img.shape[1] // 2 - (size_y // 2)
    joints_2d[0, :, :] = joints_2d[0, :, :] - start_x
    joints_2d[1, :, :] = joints_2d[1, :, :] - start_y
    img = img[:, start_y:start_y + size_y, start_x:start_x + size_x, :]
    return img, joints_2d


def get_order_joint_human(joints):
    """
    Order:
        0 RFoot, 1 RKnee, 2 RHip, 3 LHip, 4 LKnee, 5 Lfoot, 6 Hip, 7 Spine, 8 Thorax, 9 Head, 10 RHand, 11 RElbow, 12 RShoulder
        13 LShoulder, 14 LElbow, 15 LHand
    """
    new_order_joints = np.zeros((joints.shape[0], 32))
    # actual index: position in surreal, index = position in human
    # permutation = [0, 6, 1, 12, 7, 2, 4, 8, 3, 13, 5, 9, 14, 10, 11, 15, 17, 25, 18, 26, 19, 27, 16, 20]
    permutation = [3, 2, 1, 7, 8, 0, 12, 13, 15, 27, 26, 25, 17, 18, 19]
    for k, i in enumerate(permutation):
        new_order_joints[:, i] = joints[:, k]
    return new_order_joints


def get_camera_matrices(cam_location):
    """
    Adapted from https://github.com/gulvarol/bodynet/blob/f07dec69a6490ecdeb1dfd3758e76a3ca5887e5b/training/util/camerautils.lua#L47
    Args:
        cam_location: 3x1, Location of the camera in world coordinate

    Returns:
        - T Translation matrix 3x1
        - R Rotation matrix 3x3
    """
    R_world_to_cam = np.matrix("0 0 1; 0 -1 0; -1 0 0").transpose()  # 3x3
    T_world_to_cam = -1 * R_world_to_cam @ cam_location  # 3x1

    R_cam_to_cv = np.matrix("1 0 0; 0 -1 0; 0 0 -1")  # 3x3
    R = R_cam_to_cv @ R_world_to_cam  # 3x3
    T = R_cam_to_cv @ T_world_to_cam  # 3x1
    return T, R


def to_camera_coordinate(joints, T, R):
    """
    Args:
        joints: 3x24xT
        T: 3x1
        R: 3x3

    Returns:
    """
    new_joints = np.zeros_like(joints)
    for t in range(joints.shape[2]):
        new_joints[:, :, t] = (R @ joints[:, :, t]) + T
    return new_joints


def to_world_coordinate(joints, T, R):
    """
    Args:
        joints: 3x24xT
        T: 3x1
        R: 3x3

    Returns:
    """
    new_joints = np.zeros_like(joints)
    for t in range(joints.shape[2]):
        new_joints[:, :, t] = R @ (joints[:, :, t] - T)
    return new_joints


def align_3d_joints(joints):
    """
    Sets Pelvis to 0
    Args:
        joints:
    Returns: aligned joints

    """
    new_joints = joints[:]
    for t in range(joints.shape[2]):
        new_joints[:, :, t] = new_joints[:, :, t] - np.expand_dims(joints[:, 0, t], axis=1)  # pelvis at coord 0
    return new_joints


def heatmat_to_2d_joints(heatmap):
    """
    From a heatmap of the joint, returns the positions in 2D
    Args:
        heatmap: size batch x 16 x 64 x 64

    Returns: joints, size: batch x 2 x 16

    """
    joints = np.zeros((heatmap.shape[0], 2, heatmap.shape[1]))  # batch x 2 x 16
    for batch in range(heatmap.shape[0]):
        for k in range(heatmap.shape[1]):
            y, x = np.unravel_index(np.argmax(heatmap[batch, k], axis=None), heatmap[batch, k].shape)
            joints[batch, 1, k] = y / heatmap[batch, k].shape[0]
            joints[batch, 0, k] = x / heatmap[batch, k].shape[1]
    return joints


def get_joints_hourglass(joints):
    """
    Hourglass:
        0 RFoot, 1 RKnee, 2 RHip, 3 LHip, 4 LKnee, 5 Lfoot, 6 Hip, 7 Top Spine, 8 Neck, 9 Head, 10 RHand, 11 RElbow, 12 RShoulder
        13 LShoulder, 14 LElbow, 15 LHand
    Surreal:
        0 Hip, 1 LHip, 2 RHip, 3 SpineDown, 4 LKnee, 5 RKnee, 6 SpineMid, 7 LFoot, 8 RFoot, 9 SpineUp, 10 LToes,
        11 RToes, 12 Neck, 13 LChest, 14 RChest, 15 Chin, 16 LShoulder, 17 RShoulder, 18 LElbow, 19 RElbow,
        20 LWrist, 21 RWrist, 22 LHand, 23 RHand

    Args:
        joints: dim A x 24 x B
    Returns:
        dim A x 16 x B
    """
    new_joints = np.zeros((joints.shape[0], 16, joints.shape[2]))
    permutation = [8, 5, 2, 1, 4, 7, 0, 9, 12, 15, 23, 19, 17, 16, 18, 22]
    for i, k in enumerate(permutation):
        new_joints[:, i] = joints[:, k]
    return new_joints


class SurrealDataset:
    def __init__(self, path, data_type, run, dataset="cmu", video_training_output=False, frames_before=0,
                 frames_after=0):
        """
        Args:
            path: path ro root surreal data
            data_type: train, val or test
            run: run0, run1 or run2
            dataset: cmu
        """
        assert data_type in ['train', 'val', 'test'], "Surreal type must be in train, val or test."
        assert run in ['run0', 'run1', 'run2'], "Surreal run must be between run0, run1 and run2"
        assert dataset in ['cmu'], "Surreal dataset must be cmu"

        self.path = path
        self.frames_before = frames_before
        self.frames_after = frames_after
        self.video_training_output = video_training_output

        # path = os.path.abspath(path)
        root_path = os.path.join(path, dataset, data_type, run)

        self.files = []
        self.targets = []
        self.item_to_file = []
        self.first_index_for_file = {}

        for seq_name in os.listdir(root_path):
            for video in os.listdir(os.path.join(root_path, seq_name)):
                if video[-3:] == 'mp4':
                    video_path = os.path.join(root_path, seq_name, video)
                    name = video[:-4]
                    self.targets.append(os.path.join(root_path, seq_name, name + "_info.mat"))
                    self.files.append(video_path)

        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        """
        Frames are shape (T, width, height, 3)
        Args:
            item: item to load

        Returns:
            triple (frames, joints_2d, joints_3d) of numpy arrays
            n_frames = 1 + self.frames_before + self.frames_after + 1
            - frames has shape (T, channels=3, height=240, width=320)
            - joints_2d has shape (2, 16, T)
            - joints_3d has shape (3, 16, T)
            Order of the joints surreal:
                - 0 Hip, 1 LHip, 2 RHip, 3 SpineDown, 4 LKnee, 5 RKnee, 6 SpineMid, 7 LFoot, 8 RFoot, 9 SpineUp, 10 LToes,
                11 RToes, 12 Neck, 13 LChest, 14 RChest, 15 Chin, 16 LShoulder, 17 RShoulder, 18 LElbow, 19 RElbow,
                20 LWrist, 21 RWrist, 22 LHand, 23 RHand
            Order of joints at output:
                0 RFoot, 1 RKnee, 2 RHip, 3 LHip, 4 LKnee, 5 Lfoot, 6 Hip, 7 Top Spine, 8 Neck, 9 Head, 10 RHand,
                11 RElbow, 12 RShoulder, 13 LShoulder, 14 LElbow, 15 LHand
        """
        scale = 1.1
        frames = np.array([frame for frame in get_frames_from_video(self.files[item], scale=scale)]) / 255
        video_info = loadmat(self.targets[item])
        joints_2d = video_info["joints2D"] * scale  # to correspond to the image scaling
        joints_3d = video_info["joints3D"]
        camera_location = video_info["camLoc"]
        T, R = get_camera_matrices(camera_location)
        joints_3d = to_camera_coordinate(joints_3d, T, R)
        joints_3d = get_joints_hourglass(align_3d_joints(joints_3d))

        # crop to power of two for better down/up sampling in hourglass
        frames, joints_2d = center_crop(frames, joints_2d, 256, 256)
        joints_2d = get_joints_hourglass(joints_2d)

        # Normalize 2D positions
        joints_2d[0, :] = joints_2d[0, :] / frames.shape[2]
        joints_2d[1, :] = joints_2d[0, :] / frames.shape[3]

        frames = frames.transpose((0, 3, 1, 2))

        return frames, joints_2d, joints_3d


class SurrealDatasetWithVideoContinuity:
    def __init__(self, path, data_type, run, dataset="cmu", video_training_output=False, frames_before=0,
                 frames_after=0):
        self.frames_before = frames_before
        self.frames_after = frames_after
        self.video_training_output = video_training_output
        self.surreal = SurrealDataset(path, data_type, run, dataset)

    def __len__(self):
        return self.surreal.len

    def __getitem__(self, item):
        """
        Only returns n_frames = 1 + self.frames_before + self.frames_after + 1
        on the image for the learning
        Args:
            item: item to load
        Returns:
            triple (frames, joints_2d, joints_3d) of numpy arrays
            n_frames = 1 + self.frames_before + self.frames_after + 1
            - frames has shape (n_frames, channels=3, height=240, width=320)
            - joints_2d has shape (2, 24)
            - joints_3d has shape (3, 24)
        """
        frames, joints_2d, joints_3d = self.surreal[item]

        if self.video_training_output:
            cur = np.random.randint(self.frames_before, frames.shape[0] - self.frames_after - 1)
            # +1 at the end for 2 different set of video to train continuity constraint
            s = slice(cur - self.frames_before, cur + self.frames_after + 2)
            frames = frames[s]

            joints_3d = joints_3d[:, :, cur]
            joints_2d = joints_2d[:, :, cur]
        return frames, joints_2d, joints_3d
