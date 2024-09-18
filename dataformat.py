import numpy as np


def get_x_w(pose):
    return np.array([pose[0], pose[1], pose[2]])


def get_x_e(pose):
    return np.array([pose[3], pose[4], pose[5]])


def get_x_b(pose):
    return np.array([pose[6], pose[7]])


def get_x(pose):
    return (get_x_w(pose), get_x_e(pose), get_x_b(pose))
