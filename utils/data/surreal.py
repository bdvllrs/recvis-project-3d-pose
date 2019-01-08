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
        yield frame
        has_next_frame, frame = video.read()


def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


class SurrealDataset:
    def __init__(self, path, data_type, run, dataset="cmu", use_video_constraints=False, frames_before=0,
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
        self.use_video_constraints = use_video_constraints
        self.frames_before = frames_before
        self.frames_after = frames_after

        # path = os.path.abspath(path)
        root_path = os.path.join(path, dataset, data_type, run)

        self.files = []
        self.targets = []
        self.item_to_file = []
        self.first_index_for_file = {}

        self.len = 0

        for seq_name in os.listdir(root_path):
            for video in os.listdir(os.path.join(root_path, seq_name)):
                if video[-3:] == 'mp4':
                    video_path = os.path.join(root_path, seq_name, video)
                    name = video[:-4]
                    self.targets.append(os.path.join(root_path, seq_name, name + "_info.mat"))
                    self.files.append(video_path)
                    if self.use_video_constraints:
                        file_len = get_video_length(video_path)
                        self.item_to_file += [len(self.files) - 1] * (file_len - frames_before - frames_after)
                        self.first_index_for_file[len(self.files) - 1] = self.len
                        self.len += file_len - frames_before - frames_after

        if not self.use_video_constraints:
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
            - frames has shape (T, width=240, height=320, channels=3)
            - joints_2d has shape (2, 24, T)
            - joints_3d has shape (3, 24, T)
            If self.use_video_constraints is True, T = self.frames_before + self.frames_after + 1.
            Else, T = video length
        """
        if self.use_video_constraints:
            t, item = item, self.item_to_file[item]
            first_t = self.first_index_for_file[item]
            t -= first_t
        frames = np.array([frame for frame in get_frames_from_video(self.files[item])])
        video_info = loadmat(self.targets[item])
        joints_2d = video_info["joints2D"]
        joints_3d = video_info["joints3D"]
        if self.use_video_constraints:
            s = slice(self.frames_before + t, self.frames_before + t + self.frames_after + 1)
            frames = frames[s]
            joints_3d = joints_3d[:, :, s]
            joints_2d = joints_2d[:, :, s]

        return frames, joints_2d, joints_3d
