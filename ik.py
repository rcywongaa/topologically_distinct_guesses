import json
import csv
import numpy as np
from math import pi, atan2, acos, asin, sin, cos
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation as R
from dataformat import get_x
from config import planning_setting


def normalize_angle(x):
    return atan2(sin(x), cos(x))


def angle_absdiff(x, y):
    return abs(normalize_angle(y - x))


def read_trajectory_from_file(filename):
    with open(filename) as f:
        lines = f.readlines()
        x_w = []
        x_b = []
        x_e = []
        heading = []
        for line in lines:
            pose_vec = json.loads(line)
            curr_x_w, curr_x_e, curr_x_b = get_x(pose_vec)
            x_w.append(curr_x_w)
            x_e.append(curr_x_e)
            x_b.append(curr_x_b)
            heading.append(pose_vec[9])
        return np.array(x_w), np.array(x_e), np.array(x_b), np.array(heading)


"""
pan axis is -z
lift axis is +y
flex axis is -y
"""


def ik(x_b, x_w, x_e, heading, last_pan=None, last_lift=None, last_flex=None):
    if len(x_b) == 2:
        x_b = np.hstack([x_b, [planning_setting.arm_base_height]])
    pan = -normalize_angle(atan2(x_w[1] - x_b[1], x_w[0] - x_b[0]) - heading)
    if last_pan:
        pan2 = -normalize_angle(atan2(-(x_w[1] - x_b[1]), -(x_w[0] - x_b[0])) - heading)
        if angle_absdiff(last_pan, pan2) < angle_absdiff(last_pan, pan):
            pan = pan2

    # From https://stackoverflow.com/a/33920320/3177701
    bw = (x_w - x_b) / np.linalg.norm(x_w - x_b)
    lift_axis = np.array([0.0, 1.0, 0.0])
    r = R.from_rotvec([0.0, 0.0, -pan + heading])
    lift_axis = r.apply(lift_axis)
    bw0 = np.array([0.0, 0.0, 1.0])  # bw vector at home position
    lift = atan2(np.dot(np.cross(bw0, bw), lift_axis), np.dot(bw0, bw))

    real_x_w = x_b + R.from_rotvec([0.0, 0.0, -pan + heading]).apply(
        R.from_rotvec([0.0, lift, 0.0]).apply(
            np.array([0.0, 0.0, planning_setting.upperarm_length])
        )
    )

    assert_array_almost_equal(real_x_w, x_w)

    # Law of cosines
    # l1 = planning_setting.upperarm_length
    # l2 = planning_setting.forearm_length
    # l3 = np.linalg.norm(x_e - x_b)
    # flex = pi - acos((l1**2 + l2**2 - l3**2) / (2 * l1 * l2))

    we = (x_e - x_w) / np.linalg.norm(x_e - x_w)
    we0 = bw
    flex_axis = np.array([0.0, -1.0, 0.0])
    flex_axis = R.from_rotvec([0.0, 0.0, -pan + heading]).apply(flex_axis)
    flex = atan2(np.dot(np.cross(we0, we), flex_axis), np.dot(we0, we))

    return pan, lift, flex


def wrist_ik(lift, flex):
    w1 = 0
    w2 = pi + lift - flex
    w3 = 0
    return w1, w2, w3


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert to joint angles")
    parser.add_argument(
        "--input",
        help="File containing the trajectory guess",
        type=str,
        required=False,
    )
    parser.add_argument("--output", help="Output filename", type=str, required=False)

    args = parser.parse_args()

    csv_writer = None
    if args.output:
        print(f"Writing to {args.output}")
        output_file = open(args.output, "w")
        csv_writer = csv.writer(output_file, delimiter=",")

    x_w_traj, x_e_traj, x_b_traj, heading_traj = read_trajectory_from_file(args.input)

    last_pan = None
    last_lift = None
    last_flex = None
    for curr_x_w, curr_x_b, curr_x_e, curr_heading in zip(
        x_w_traj, x_b_traj, x_e_traj, heading_traj
    ):
        pan, lift, yaw = ik(
            curr_x_b, curr_x_w, curr_x_e, curr_heading, last_pan, last_lift, last_flex
        )

        last_pan = pan
        last_lift = lift
        last_yaw = yaw
        output = [curr_x_b[0], curr_x_b[1], curr_heading, pan, lift, yaw, 0.0, 0.0, 0.0]
        if csv_writer is not None:
            print(f"Writing row: {output}")
            csv_writer.writerow(output)
        else:
            print(output)

    if csv_writer is not None:
        output_file.flush()
        output_file.close()
