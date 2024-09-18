"""
This program serves to verify the equation defining the circle
traced out by the elbow when keeping the base and end effector fixed
"""

import numpy as np
from math import sqrt, sin, cos, pi, atan2, acos
import time
from pydrake.geometry import (
    Rgba,
    Sphere,
    StartMeshcat,
)
from pydrake.math import RigidTransform


def Matrix(arg):
    return np.array(arg)


def calc_elbow_position_6dof(l1, l2, x_b, x_e, is_up):
    z = np.array([0.0, 0.0, 1.0])
    A = np.array([x_b[0], x_b[1], x_b[2]])
    B = np.array(x_e)
    l3 = np.linalg.norm(A - B)
    B_proj = np.array([B[0], B[1], A[2]])
    AB_proj = B_proj - A
    AB_proj_norm = np.linalg.norm(AB_proj)

    phi1 = atan2(B[2] - A[2], AB_proj_norm)
    phi2 = acos((l1**2 + l3**2 - l2**2) / (2.0 * l1 * l3))
    if not is_up:
        phi2 = -phi2
    return A + (AB_proj / AB_proj_norm * cos(phi1 + phi2) + z * sin(phi1 + phi2)) * l1


def calc_elbow_position_7dof(l1, l2, x_b, x_e, n):
    delta = x_e - x_b
    # d = delta.norm()
    d = np.linalg.norm(delta)
    # phi = delta.normalized()
    phi = delta / d
    phi_1 = phi[0]
    phi_2 = phi[1]
    phi_3 = phi[2]
    # r = sqrt(1 - d**2 / 4)
    r = sqrt(l2**2 - (l2**2 + d**2 - l1**2) ** 2 / (4 * d**2))
    l1_perp = sqrt(l1**2 - r**2)
    return Matrix(
        [
            l1_perp * (1 + (-(phi_2**2) - phi_3**2) / (phi_1 + 1))
            - phi_2 * r * cos(n)
            - phi_3 * r * sin(n)
            + x_b[0],
            l1_perp * phi_2
            - (phi_2 * phi_3 * r * sin(n)) / (phi_1 + 1)
            + r * (-(phi_2**2) / (phi_1 + 1) + 1) * cos(n)
            + x_b[1],
            l1_perp * phi_3
            - (phi_2 * phi_3 * r * cos(n)) / (phi_1 + 1)
            + r * (-(phi_3**2) / (phi_1 + 1) + 1) * sin(n)
            + x_b[2],
        ]
    )


def show_indicators(l1, l2, x_b, x_e, n):
    radius = 0.1
    meshcat.SetTransform("x_b", RigidTransform(x_b))
    meshcat.SetObject("x_b", Sphere(radius), Rgba(0, 1, 0, 1))
    meshcat.SetTransform("x_e", RigidTransform(x_e))
    meshcat.SetObject("x_e", Sphere(radius), Rgba(0, 0, 1, 1))

    p_Wc = calc_elbow_position(l1, l2, x_b, x_e, n)
    p_Wc = p_Wc.reshape((3, 1))
    meshcat.SetTransform("c", RigidTransform(p_Wc))
    meshcat.SetObject("c", Sphere(radius), Rgba(1, 0, 0, 1))


if __name__ == "__main__":
    meshcat = StartMeshcat()

    random_range = 0.7
    offset_range = 2
    while True:
        offset = offset_range * (np.random.random((3, 1)) - offset_range / 2)
        x_b = random_range * (np.random.random((3, 1)) - random_range / 2) + offset
        x_e = random_range * (np.random.random((3, 1)) - random_range / 2) + offset
        # x_b = np.array([[0], [0], [0]])
        # x_e = np.array([[0.0], [0.5], [0]])
        for n in np.arange(0, 2 * pi, 0.02):
            show_indicators(0.3, 0.4, x_b, x_e, n)
            time.sleep(0.05)
        time.sleep(2)
