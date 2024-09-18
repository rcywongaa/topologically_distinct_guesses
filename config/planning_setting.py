import numpy as np
import math
from math import pi
from pydrake.math import RigidTransform
from pydrake.geometry import Rgba, Sphere, Cylinder, Box
import importlib
from itertools import product
import uuid
from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
)
from pydrake.math import (
    ComputeNumericalGradient,
    NumericalGradientMethod,
    NumericalGradientOption,
)
from functools import reduce

import pybullet as p

p.connect(p.DIRECT)
p.resetSimulation()

max_distance = 100
# scene_name = "simple_scene"
scene_name = "bar_table_scene"
# scene_name = "rosie_scene"
# wavy = False
wavy = True

try:
    from . import calc_elbow_position
    scene = importlib.import_module("." + scene_name, "config")
except ImportError:  # Allow running as script
    import calc_elbow_position
    scene = importlib.import_module(scene_name)


# Disable SIGINT handler so that rust can receive it
# This doesn't affect running this in python
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

# REQUIRED
sphere_obstacle_specs = scene.sphere_obstacle_specs
aabb_obstacle_specs = scene.aabb_obstacle_specs
cylinder_obstacle_specs = scene.cylinder_obstacle_specs


class SphereObstacle:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius
        self.sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)

    # def is_in_obstacle(self, sample):
    #     return l2_norm_sq_expr(sample - self.center) < self.radius**2

    def show(self, meshcat, name):
        meshcat.SetTransform(name, RigidTransform(self.center))
        meshcat.SetObject(name, Sphere(self.radius), Rgba(0.2, 0.2, 0.2, 1))

    def dist(self, sample, col):
        pts = p.getClosestPoints(
            bodyA=-1,
            bodyB=-1,
            distance=max_distance,
            collisionShapeA=col,
            collisionShapePositionA=sample,
            collisionShapeB=self.sphere_col,
            collisionShapePositionB=self.center,
        )
        # Returns empty list if greater than max_distance
        if len(pts) == 0:
            d = max_distance
        else:
            d = pts[0][8]
        return np.array([d])


class CylinderObstacle:
    def __init__(self, center, radius_height):
        self.center = np.array(center)
        self.radius_height = radius_height
        self.radius = radius_height[0]
        self.height = radius_height[1]
        self.cylinder_col = p.createCollisionShape(
            p.GEOM_CYLINDER, radius=self.radius, height=self.height
        )

    def dist(self, sample, col):
        pts = p.getClosestPoints(
            bodyA=-1,
            bodyB=-1,
            distance=max_distance,
            collisionShapeA=col,
            collisionShapePositionA=sample,
            collisionShapeB=self.cylinder_col,
            collisionShapePositionB=self.center,
        )
        # Returns empty list if greater than max_distance
        if len(pts) == 0:
            d = max_distance
        else:
            d = pts[0][8]
        return np.array([d])

    def show(self, meshcat, name):
        meshcat.SetTransform(name, RigidTransform(self.center))
        meshcat.SetObject(
            name, Cylinder(self.radius, self.height), Rgba(0.2, 0.2, 0.2, 1.0)
        )


class AabbObstacle:
    def __init__(self, center, dims):
        self.center = np.array(center)
        self.dims = np.array(dims)
        """ pybullet setup """
        self.box_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.dims[0] / 2, self.dims[1] / 2, self.dims[2] / 2],
        )
        """"""

    def show(self, meshcat, name):
        meshcat.SetTransform(name, RigidTransform(self.center))
        meshcat.SetObject(
            name,
            Box(self.dims[0], self.dims[1], self.dims[2]),
            Rgba(0.2, 0.2, 0.2, 1.0),
        )

    def dist(self, sample, col):
        """distance_3d"""
        # sphere = colliders.Sphere(sample, scene.link_radius)
        # box_transform = np.eye(4)
        # box_transform[:3, 3] = self.center
        # box = colliders.Box(box_transform, self.dims)
        # return np.array([gjk.gjk_distance(sphere, box)[0]])
        """FCL does not report signed distance"""
        # sphere = fcl.CollisionObject(fcl.Sphere(scene.link_radius), fcl.Transform())
        # sphere.setTranslation(sample)
        # box = fcl.CollisionObject(fcl.Box(*self.dims), fcl.Transform())
        # box.setTranslation(self.center)
        # req = fcl.DistanceRequest()
        # res = fcl.DistanceResult()
        # ret = fcl.distance(sphere, box, req, res)
        # if ret < self.min_dist:
        #     self.min_dist = ret
        #     print(f"min dist = {self.min_dist}")
        #     print(f"sample = {sample}")
        #     print(
        #         f"res b1: {res.b1}, b2: {res.b2}, nearest point: {res.nearest_points}, o1: {res.o1}, o2: {res.o2}"
        #     )
        # return np.array([ret])
        """ pybullet attempt """
        pts = p.getClosestPoints(
            bodyA=-1,
            bodyB=-1,
            distance=max_distance,
            collisionShapeA=col,
            collisionShapePositionA=sample,
            collisionShapeB=self.box_col,
            collisionShapePositionB=self.center,
        )
        # Returns empty list if greater than max_distance
        if len(pts) == 0:
            d = max_distance
        else:
            d = pts[0][8]
        return np.array([d])


# To handle arbitrary obstacles:
# https://github.com/RobotLocomotion/drake/issues/9138
obstacles = (
    [SphereObstacle(*spec) for spec in sphere_obstacle_specs]
    + [CylinderObstacle(*spec) for spec in cylinder_obstacle_specs]
    + [AabbObstacle(*spec) for spec in aabb_obstacle_specs]
)


def show_obstacles(meshcat):
    for idx, obs in enumerate(obstacles):
        name = f"obstacle{idx}"
        obs.show(meshcat, name)


def l2_norm_sq_expr(expr):
    return expr.T @ expr

"""
No longer used due to poor performance
"""
# def is_in_obstacle(sample):
#     for obs in obstacles:
#         if obs.is_in_obstacle(sample):
#             return True
#     return False
""""""


class LineTrajectory:
    def __init__(self, begin, end):
        self.begin = np.array(begin)
        self.end = np.array(end)

    def get(self, t):
        return self.begin + (self.end - self.begin) * t


class SinYTrajectory:
    def __init__(self, begin, end, amplitude, period):
        self.begin = np.array(begin)
        self.end = np.array(end)
        self.amplitude = amplitude
        self.period = period

    def get(self, t):
        return (
            self.begin
            + (self.end - self.begin) * t
            + np.array([0.0, self.amplitude, 0.0]) * math.sin(t * 2 * pi / self.period)
        )


def get_eef_position(t):
    assert t <= 1.0
    assert t >= 0.0
    return eef_trajectory.get(t)


# REQUIRED
"""
Changing arm lengths seems to require increasing obstacle sizes to retain performance
"""
# forearm_length = 0.2084 + 0.1059
# upperarm_length = 0.41
# arm_base_height = 0.6438  # 0.1284 + 0.1564 + 0.359
forearm_length = scene.forearm_length
upperarm_length = scene.upperarm_length
arm_base_height = scene.arm_base_height
mobile_base_height = scene.mobile_base_height

link_radius = scene.link_radius
base_radius = scene.base_radius
link_col = p.createCollisionShape(p.GEOM_SPHERE, radius=scene.link_radius)
# print(f"Link col: {link_col}")
base_col = p.createCollisionShape(
    p.GEOM_CYLINDER, radius=scene.base_radius, height=scene.mobile_base_height
)
# print(f"Base col: {base_col}")

# eef_start = [-1.0, 0.0, 0.35 + arm_base_height]
# eef_end = [1.0, 0.0, 0.35 + arm_base_height]
# eef_start = [-1.0, 0.0, 0.5]
# eef_end = [1.0, 0.0, 0.5]
eef_start = scene.eef_start
eef_end = scene.eef_end
if wavy:
    eef_trajectory = SinYTrajectory(eef_start, eef_end, 0.08, 0.25)
else:
    eef_trajectory = LineTrajectory(eef_start, eef_end)

# initial_x_b = np.array([-1.0, 0.1, 0.0])
initial_x_b = scene.initial_x_b
initial_theta = scene.initial_theta
initial_x_e = get_eef_position(0.0)
initial_x_w = calc_elbow_position.calc_elbow_position_6dof(
    upperarm_length, forearm_length, initial_x_b, initial_x_e, True
)
# final_x_b = np.array([1.0, 0.1, 0.0])
final_x_b = scene.final_x_b
final_theta = scene.final_theta
final_x_e = get_eef_position(1.0)
final_x_w = calc_elbow_position.calc_elbow_position_6dof(
    upperarm_length, forearm_length, final_x_b, final_x_e, True
)
num_paths = scene.num_paths
T = scene.T
dt = scene.dt
buffer = scene.buffer


# Computes the signed angle from vector a to vector b
# along axis given by n
def get_signed_angle(a, b, n):
    n = np.array(n) / np.linalg.norm(n)
    return math.atan2(np.dot(np.cross(a, b), n), np.dot(a, b))


def calc_pose(x_b, x_w, x_e):
    base_x = float(x_b[0])
    base_y = float(x_b[1])
    dx_w = x_w - x_b
    dx_w_dir = dx_w / np.linalg.norm(dx_w)
    shoulder_pan = math.atan2(dx_w[1], dx_w[0])
    shoulder_lift = math.atan2(dx_w[2], math.sqrt(dx_w[0] ** 2 + dx_w[1] ** 2))
    dx_e = x_e - x_w
    elbow_flex_axis = [-dx_w[1], dx_w[0], 0]
    elbow_flex_axis = elbow_flex_axis / np.linalg.norm(elbow_flex_axis)
    # https://stackoverflow.com/a/33920320/3177701
    elbow_flex = get_signed_angle(dx_w, dx_e, elbow_flex_axis)

    proj_dx_e = np.dot(dx_e, dx_w_dir) * dx_w_dir
    perp_dx_e = dx_e - proj_dx_e
    proj_dx_e_0 = forearm_length * math.cos(elbow_flex) * dx_w_dir
    perp_dx_e_0 = dx_e - proj_dx_e_0
    shoulder_roll = get_signed_angle(perp_dx_e_0, perp_dx_e, dx_w_dir)
    return (base_x, base_y, shoulder_lift, shoulder_pan, shoulder_roll, elbow_flex)


def to_rosparam_yaml():
    import yaml

    sphere_obstacle_names = []
    # sphere_centers = []
    sphere_centers = {}
    sphere_radii = []
    for idx, sphere_obstacle_spec in enumerate(sphere_obstacle_specs):
        name = "sphere_obstacle" + str(idx)
        sphere_obstacle_names.append(name)
        # sphere_centers.append(list(sphere_obstacle_spec[0]))
        # HACK due to https://github.com/ros2/rcl/issues/463
        # sphere_centers["x" + str(idx)] = sphere_obstacle_spec[0][0]
        # sphere_centers["y" + str(idx)] = sphere_obstacle_spec[0][1]
        # sphere_centers["z" + str(idx)] = sphere_obstacle_spec[0][2]
        sphere_centers[str(idx)] = {
            "x": sphere_obstacle_spec[0][0],
            "y": sphere_obstacle_spec[0][1],
            "z": sphere_obstacle_spec[0][2],
        }
        sphere_radii.append(sphere_obstacle_spec[1])

    cylinder_obstacle_names = []
    cylinder_centers = {}
    cylinder_radii = []
    cylinder_heights = []
    for idx, cylinder_obstacle_spec in enumerate(cylinder_obstacle_specs):
        name = "cylinder_obstacle" + str(idx)
        cylinder_obstacle_names.append(name)
        cylinder_centers[str(idx)] = {
            "x": cylinder_obstacle_spec[0][0],
            "y": cylinder_obstacle_spec[0][1],
            "z": cylinder_obstacle_spec[0][2],
        }
        cylinder_radii.append(cylinder_obstacle_spec[1][0])
        cylinder_heights.append(cylinder_obstacle_spec[1][1])

    aabb_obstacle_names = []
    aabb_centers = {}
    aabb_dims = {}
    for idx, aabb_obstacle_spec in enumerate(aabb_obstacle_specs):
        name = "aabb_obstacle" + str(idx)
        aabb_obstacle_names.append(name)
        aabb_centers[str(idx)] = {
            "x": aabb_obstacle_spec[0][0],
            "y": aabb_obstacle_spec[0][1],
            "z": aabb_obstacle_spec[0][2],
        }
        aabb_dims[str(idx)] = {
            "x": aabb_obstacle_spec[1][0],
            "y": aabb_obstacle_spec[1][1],
            "z": aabb_obstacle_spec[1][2],
        }

    (
        initial_base_x,
        initial_base_y,
        initial_shoulder_lift,
        initial_shoulder_pan,
        initial_shoulder_roll,
        initial_elbow_flex,
    ) = calc_pose(initial_x_b, initial_x_w, initial_x_e)
    initial_positions = {
        "base_x": initial_base_x,
        "base_y": initial_base_y,
        "elbow_flex": initial_elbow_flex,
        "shoulder_lift": initial_shoulder_lift,
        "shoulder_pan": initial_shoulder_pan,
        "shoulder_roll": initial_shoulder_roll,
        "eef_spherical_x": 0.0,
        "eef_spherical_y": 0.0,
        "eef_spherical_z": 0.0,
    }
    (
        final_base_x,
        final_base_y,
        final_shoulder_lift,
        final_shoulder_pan,
        final_shoulder_roll,
        final_elbow_flex,
    ) = calc_pose(final_x_b, final_x_w, final_x_e)
    final_positions = {
        "base_x": final_base_x,
        "base_y": final_base_y,
        "elbow_flex": final_elbow_flex,
        "shoulder_lift": final_shoulder_lift,
        "shoulder_pan": final_shoulder_pan,
        "shoulder_roll": final_shoulder_roll,
        "eef_spherical_x": 0.0,
        "eef_spherical_y": 0.0,
        "eef_spherical_z": 0.0,
    }

    parameters = {
        "eef_start": eef_start,
        "eef_end": eef_end,
        "forearm_length": forearm_length,
        "upperarm_length": upperarm_length,
        "initial_x_b": initial_x_b.tolist(),
        "initial_theta": initial_theta,
        "initial_x_e": initial_x_e.tolist(),
        "initial_x_w": initial_x_w.tolist(),
        "final_x_b": final_x_b.tolist(),
        "final_theta": final_theta,
        "final_x_e": final_x_e.tolist(),
        "final_x_w": final_x_w.tolist(),
        "num_paths": num_paths,
        "initial_positions": initial_positions,
        "final_positions": final_positions,
    }

    # ROS2 doesn't support empty lists: https://github.com/ros2/rclcpp/issues/1955
    if sphere_obstacle_names:
        parameters.update(
            {
                "sphere_obstacles": sphere_obstacle_names,
                "sphere_obstacle_centers": sphere_centers,
                "sphere_obstacle_radii": sphere_radii,
            }
        )

    if cylinder_obstacle_names:
        parameters.update(
            {
                "cylinder_obstacles": cylinder_obstacle_names,
                "cylinder_obstacle_centers": cylinder_centers,
                "cylinder_obstacle_radii": cylinder_radii,
                "cylinder_obstacle_heights": cylinder_heights,
            }
        )
    if aabb_obstacle_names:
        parameters.update(
            {
                "aabb_obstacles": aabb_obstacle_names,
                "aabb_obstacle_centers": aabb_centers,
                "aabb_obstacle_dims": aabb_dims,
            }
        )

    param = {"/**": {"ros__parameters": parameters}}
    with open("planning_setting.yaml", "w") as output_file:
        yaml.dump(param, output_file)

    with open("initial_positions.yaml", "w") as output_file:
        yaml.dump({"initial_positions": initial_positions}, output_file)


if __name__ == "__main__":
    to_rosparam_yaml()
