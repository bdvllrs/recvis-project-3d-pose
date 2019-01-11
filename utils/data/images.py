import os

import cv2
import numpy as np

from utils.data import img as I


class ImageSequence:
    def __init__(self, path):
        self.path = path

        root_path = os.path.abspath(os.path.join(os.curdir, path))

        self.frames = []
        for frame in sorted(os.listdir(root_path)):
            self.frames.append(os.path.join(root_path, frame))

        self.input_res = 256
        self.output_res = 32
        self.n_joints = 16
        self.len = len(self.frames)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        path = self.frames[item]
        img = cv2.imread(path)
        c = np.array([img.shape[0] // 2, img.shape[1] // 2])
        c[1] = c[1] - 80
        c[0] = c[0] + 80
        s = 2 * 200
        r = 0

        inp = I.Crop(img, c, s, r, self.input_res) / 256.

        return inp
