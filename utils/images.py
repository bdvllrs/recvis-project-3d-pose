from utils.data import heatmat_to_2d_joints, get_order_joint_human
import numpy as np


def get_2d_joints(hg_model, images, human_order=True):
    """
    Args:
        human_order: if True, permute dimension like in human
        hg_model:
        images: batch of images size: batch x 3 x width x height
    Returns: joints 2d in same order as human size batch x n_joints x 3
    6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10
    """
    predicted_pose = hg_model(images)[-1].detach().cpu().numpy()
    joints_2d = heatmat_to_2d_joints(predicted_pose)
    if human_order:
        new_joints = np.zeros((joints_2d.shape[0], 2, 32))
        for k in range(joints_2d.shape[0]):
            new_joints[k] = get_order_joint_human(joints_2d[k])
        return new_joints[:, :, [0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27]].transpose((0, 2, 1)).reshape(joints_2d.shape[0], -1)
    return joints_2d.transpose((0, 2, 1)).reshape(joints_2d.shape[0], -1)


def get_all_32joints_old(joints_cur):
    """

    Args:
        joints_cur:

    Returns:
    """
    joints = np.zeros((joints_cur.shape[0], 32, joints_cur.shape[2]))
    permutation = np.array([0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27])
    for t in range(joints_cur.shape[0]):
        for k in range(joints_cur.shape[1]):
            joints[t, permutation[k]] = joints_cur[t, k]
    return joints.reshape(joints_cur.shape[0], 32*joints_cur.shape[2])


def get_all_32joints(data, dim, dimensions_to_ignore):
    T = data.shape[0]  # Batch size
    D = 32*dim  # Dimensionality

    orig_data = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = np.array([dim for dim in range(D)
                                  if dim not in dimensions_to_ignore])

    orig_data[:, dimensions_to_use] = data
    return orig_data
