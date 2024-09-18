import json
from pathlib import Path
from time import sleep
from math import pi, atan2, acos, asin, sin, cos
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation as R
import uuid


from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.visualization import AddDefaultVisualization
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.all import MeshcatVisualizer
from pydrake.geometry import (
    Rgba,
    Sphere,
    StartMeshcat,
)

from config import planning_setting
from dataformat import get_x

fetch_description_name = "fetch_description"
turtlebot4_description_name = "turtlebot4_description"
kortex_description_name = "kortex_description"


def wrap_angle_nearest(x, initial):
    diff = abs(x - initial)
    while True:
        x1 = x + 2 * pi
        diff1 = abs(x - x1)
        x2 = x - 2 * pi
        diff2 = abs(x - x2)
        if diff1 < diff2:
            if diff1 < diff:
                x = x1
                diff = diff1
            else:
                break
        else:
            if diff2 < diff:
                x = x2
                diff = diff2
            else:
                break
    return x


def normalize_angle(x):
    return atan2(sin(x), cos(x))


def angle_absdiff(x, y):
    return abs(normalize_angle(y - x))


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


# height_offset = 0.359 + 0.156 + 0.128 # Previous hack for simple_scene


def show_pose(meshcat, x_b, x_w, x_e, theta, is_persist=False):
    radius = 0.025
    link_color = Rgba(0.0, 0.0, 0.0, 0.25)
    link_width = 1  # pixels

    x_b_name = "x_b"
    x_w_name = "x_w"
    x_e_name = "x_e"
    heading_indicator_name = "heading_indicator"
    upperarm_name = "upperarm"
    forearm_name = "forearm"
    if is_persist:
        x_b_name = str(uuid.uuid4())
        x_w_name = str(uuid.uuid4())
        x_e_name = str(uuid.uuid4())
        heading_indicator_name = str(uuid.uuid4())
        upperarm_name = str(uuid.uuid4())
        forearm_name = str(uuid.uuid4())

    if len(x_b) == 2:
        x_b = np.hstack([x_b, planning_setting.arm_base_height])
    meshcat.SetTransform(x_b_name, RigidTransform(x_b))
    meshcat.SetObject(x_b_name, Sphere(radius), Rgba(1, 0, 0, 0.5))
    heading_indicator_position = (
        RigidTransform(x_b)
        .multiply(RigidTransform(rpy=RollPitchYaw([0, 0, theta]), p=[0, 0, 0]))
        .multiply(RigidTransform([radius, 0, 0]))
    )

    meshcat.SetTransform(
        heading_indicator_name, RigidTransform(heading_indicator_position)
    )
    meshcat.SetObject(heading_indicator_name, Sphere(0.5 * radius), Rgba(1, 1, 1, 0.5))

    # x_w[2] += height_offset
    meshcat.SetTransform(x_w_name, RigidTransform(x_w))
    meshcat.SetObject(x_w_name, Sphere(radius), Rgba(0, 1, 0, 0.5))

    # x_e[2] += height_offset
    meshcat.SetTransform(x_e_name, RigidTransform(x_e))
    meshcat.SetObject(x_e_name, Sphere(radius), Rgba(0, 0, 1, 0.5))

    meshcat.SetLine(
        upperarm_name,
        np.hstack(
            [x_b.reshape((-1, 1)), x_w.reshape((-1, 1))],
        ),
        link_width,
        link_color,
    )
    meshcat.SetLine(
        forearm_name,
        np.hstack(
            [x_w.reshape((-1, 1)), x_e.reshape((-1, 1))],
        ),
        link_width,
        link_color,
    )


def display_trajectory(x_w_traj, x_b_traj, x_e_traj, heading_traj):
    meshcat = StartMeshcat()
    meshcat.SetCameraPose(
        camera_in_world=[-1.5, -1.0, 2.5], target_in_world=[0.0, 0.0, 0.0]
    )
    planning_setting.show_obstacles(meshcat)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder=builder, time_step=0.01)
    parser = Parser(plant=plant)
    # parser.package_map().PopulateFromRosPackagePath()
    parser.package_map().PopulateFromEnvironment("AMENT_PREFIX_PATH")

    print(f"Package map: {parser.package_map()}")
    # mobile_base_urdf = (
    #     parser.package_map().GetPath(fetch_description_name) + "/robots/freight.urdf"
    # )
    mobile_base_urdf = (
        parser.package_map().GetPath(turtlebot4_description_name)
        + "/urdf/standard/turtlebot4.urdf"
    )
    mobile_base_model = parser.AddModelsFromString(
        Path(mobile_base_urdf).read_text(), "urdf"
    )[0]
    # jaco_model = parser.AddModelsFromUrl(
    #     "package://drake_models/jaco_description/urdf/j2s7s300_sphere_collision.urdf"
    # )[0]
    arm_model = parser.AddModelsFromString(
        Path(
            parser.package_map().GetPath(kortex_description_name)
            + "/arms/gen3/6dof/urdf/GEN3-6DOF_NO-VISION_URDF_ARM_V01.urdf"
        ).read_text(),
        "urdf",
    )[0]

    mobile_base_base_link = plant.GetFrameByName("base_link", mobile_base_model)
    mobile_base_world_link = plant.GetFrameByName("world_base", mobile_base_model)
    # jaco_base_link = plant.GetFrameByName("base", jaco_model)
    arm_base_link = plant.GetFrameByName("base_link", arm_model)

    plant.WeldFrames(
        plant.world_frame(), mobile_base_world_link, RigidTransform([0.0, 0.0, 0.0])
    )
    # plant.WeldFrames(fetch_base_link, jaco_base_link, RigidTransform([0.0, 0.0, 0.359]))
    plant.WeldFrames(
        mobile_base_base_link, arm_base_link, RigidTransform([0.0, 0.0, 0.359])
    )

    plant.Finalize()

    visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()

    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(diagram_context)

    visualizer.StartRecording()
    t = 0.0
    last_pan = 0.0
    for curr_x_w, curr_x_b, curr_x_e, curr_heading in zip(
        x_w_traj, x_b_traj, x_e_traj, heading_traj
    ):
        # print(f"x_w: {curr_x_w}, x_b: {curr_x_b}, heading: {curr_heading}")
        diagram_context.SetTime(t)
        (pan, lift, elbow_flex) = ik(
            x_b=curr_x_b,
            x_w=curr_x_w,
            x_e=curr_x_e,
            heading=curr_heading,
            last_pan=last_pan,
        )
        # print(f"Heading: {curr_heading}, pan: {pan}, lift: {lift}, flex: {elbow_flex}")
        (w1, w2, w3) = wrist_ik(lift=lift, flex=elbow_flex)
        positions = np.zeros((plant.num_positions(), 1))
        pan_joint_idx = plant.GetJointByName("joint_1", arm_model).position_start()
        lift_joint_idx = plant.GetJointByName("joint_2", arm_model).position_start()
        elbow_flex_joint_idx = plant.GetJointByName(
            "joint_3", arm_model
        ).position_start()
        base_x_idx = plant.GetJointByName("x", mobile_base_model).position_start()
        base_y_idx = plant.GetJointByName("y", mobile_base_model).position_start()
        base_heading_idx = plant.GetJointByName(
            "heading", mobile_base_model
        ).position_start()
        w1_idx = plant.GetJointByName("joint_4", arm_model).position_start()
        w2_idx = plant.GetJointByName("joint_5", arm_model).position_start()
        w3_idx = plant.GetJointByName("joint_6", arm_model).position_start()
        positions[pan_joint_idx, 0] = pan
        positions[lift_joint_idx, 0] = lift
        positions[elbow_flex_joint_idx, 0] = elbow_flex
        positions[w1_idx, 0] = w1
        positions[w2_idx, 0] = w2
        positions[w3_idx, 0] = w3
        positions[base_x_idx, 0] = curr_x_b[0]
        positions[base_y_idx, 0] = curr_x_b[1]
        positions[base_heading_idx, 0] = curr_heading
        plant.SetPositions(plant_context, positions)
        diagram.ForcedPublish(diagram_context)
        t += 0.2
        last_pan = pan

        show_pose(
            meshcat,
            x_b=curr_x_b,
            x_w=curr_x_w,
            x_e=curr_x_e,
            theta=curr_heading,
            is_persist=True,
        )

    visualizer.StopRecording()
    visualizer.PublishRecording()

    while True:
        sleep(0.5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize trajectory with real robot.  Remember to source the ros2_ws first."
    )
    parser.add_argument(
        "--filename",
        help="File containing the trajectory guess",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    x_w_traj, x_e_traj, x_b_traj, heading_traj = read_trajectory_from_file(
        args.filename
    )
    display_trajectory(
        x_w_traj=x_w_traj,
        x_b_traj=x_b_traj,
        x_e_traj=x_e_traj,
        heading_traj=heading_traj,
    )
