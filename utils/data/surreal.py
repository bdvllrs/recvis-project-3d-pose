import numpy as np
import cv2
import os
from scipy.io import loadmat


def get_frames_from_video(video_path):
    """
    Get frame generator from video file path
    Args:
        video_path: path of the video

    Returns: generator of the frames
    """
    video = cv2.VideoCapture(video_path)
    has_next_frame, frame = video.read()
    while has_next_frame:
        # *1.1 on the size so that we can crop to (256, 256)
        frame = cv2.resize(frame, dsize=(352, 264), interpolation=cv2.INTER_CUBIC)
        yield frame
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


class SurrealDataset:
    def __init__(self, path, data_type, run, dataset="cmu", frames_before=0, frames_after=0):
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
            - frames has shape (n_frames, channels=3, height=240, width=320)
            - joints_2d has shape (2, 24, 2=T=2 frames)
            - joints_3d has shape (3, 24, 2=T=2 frames)
            Order of the joints:
                - Hip, LHip, RHip, SpineDown, LKnee, RKnee, SpineMid, LFoot, RFoot, SpineUp, LToes, RToes, Neck, LChest,
                RChest, Chin, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHand, RHand
        """
        frames = np.array([frame for frame in get_frames_from_video(self.files[item])])
        video_info = loadmat(self.targets[item])
        joints_2d = video_info["joints2D"] * 1.1  # to correspond to the image scaling
        joints_3d = video_info["joints3D"]
        # crop to power of two for better down/up sampling in hourglass
        frames, joints_2d = center_crop(frames, joints_2d, 256, 256)
        cur = np.random.randint(self.frames_before, frames.shape[0] - self.frames_after - 1)
        # +1 at the end for 2 different set of video to train continuity constraint
        s = slice(cur - self.frames_before, cur + self.frames_after + 2)
        frames = frames[s]

        joints_2d = joints_2d[:, :, cur]
        # Normalize 2D positions
        joints_2d[0, :] = joints_2d[0, :] / frames.shape[2]
        joints_2d[1, :] = joints_2d[0, :] / frames.shape[3]
        joints_3d = joints_3d[:, :, cur]

        frames = frames.transpose((0, 3, 1, 2))

        return frames, joints_2d, joints_3d
