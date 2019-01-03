"""Utilities to deal with the cameras of human3.6m
Copied from @una-dinosauria on Github
Author: Julieta Martinez
"""

from __future__ import division

import numpy as np


def load_camera_params(cameras, path):
    """
    Load h36m camera parameters
    Args:
        cameras: hdf5 open file with h36m cameras data
        path: path or key inside hf to the camera we are interested in
    Returns:
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
        name: String with camera id
    """

    R = cameras[path.format('R')][:]
    R = R.T

    T = cameras[path.format('T')][:]
    f = cameras[path.format('f')][:]
    c = cameras[path.format('c')][:]
    k = cameras[path.format('k')][:]
    p = cameras[path.format('p')][:]

    name = cameras[path.format('Name')][:]
    name = "".join([chr(item) for item in name])

    return R, T, f, c, k, p, name


def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
      P: Nx3 points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
      f: (scalar) Camera focal length
      c: 2x1 Camera center
      k: 3x1 Camera radial distortion coefficients
      p: 2x1 Camera tangential distortion coefficients
    Returns
      Proj: Nx2 points in pixel space
      D: 1xN depth of each point in camera space
      radial: 1xN radial distortion per point
      tan: 1xN tangential distortion per point
      r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


def world_to_camera_frame(P, R, T):
    """
    Convert points from world to camera coordinates
    Args
      P: Nx3 3d points in world coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
    Returns
      X_cam: Nx3 3d points in camera coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.dot(P.T - T)  # rotate and translate

    return X_cam.T


def camera_to_world_frame(P, R, T):
    """Inverse of world_to_camera_frame
    Args
      P: Nx3 points in camera coordinates
      R: 3x3 Camera rotation matrix
      T: 3x1 Camera translation parameters
    Returns
      X_cam: Nx3 points in world coordinates
    """

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    X_cam = R.T.dot(P.T) + T  # rotate and translate

    return X_cam.T


def transform_world_to_camera(poses_set, cams, joint_names, ncams=4):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args:
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Returns:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):

        subj, action, seqname = t3dk
        t3d_world = poses_set[t3dk]

        for c in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, c + 1)]
            camera_coord = world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape(camera_coord, [-1, len(joint_names) * 3])
            # camera_coord = t3d_world

            sname = seqname[:-3] + "." + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t3d_camera[(subj, action, sname)] = camera_coord

    return t3d_camera


def project_to_cameras(poses_set, cams, joint_names, ncams=4):
    """
    Project 3d poses using camera parameters
    Args:
        poses_set: dictionary with 3d poses
        cams: dictionary with camera parameters
        ncams: number of cameras per subject
        joint_names: List of joint names
    Returns:
        t2d: dictionary with 2d poses
    """
    t2d = {}

    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]

        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam + 1)]
            pts2d, _, _, _, _ = project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)

            pts2d = np.reshape(pts2d, [-1, len(joint_names) * 2])
            sname = seqname[:-3] + "." + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t2d[(subj, a, sname)] = pts2d

    return t2d
