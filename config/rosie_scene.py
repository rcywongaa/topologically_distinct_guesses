import numpy as np
from math import pi

sphere_obstacle_specs = []

aabb_obstacle_specs = []

cylinder_obstacle_specs = [
    (
        (-0.5, 0.0, 0.1),
        (0.1, 0.2),
    ),
    (
        (0.5, 0.0, 0.1),
        (0.1, 0.2),
    ),
]

link_radius = 0.05
base_radius = 0.25

forearm_length = 0.325
upperarm_length = 0.325
mobile_base_height = 0.206
arm_base_height = mobile_base_height + 0.075  # from rosie.urdf
eef_height = 0.5
eef_start = [-1.0, 0.0, eef_height]
eef_end = [1.0, 0.0, eef_height]
initial_theta = 0.01
initial_x_b = np.array([-1.0, 0.05, arm_base_height])
final_theta = 0.01
final_x_b = np.array([1.0, 0.05, arm_base_height])

num_paths = 4

T = 150
dt = 0.2

# This gives some time for robot to orient itself before moving eef.
# Avoids requiring extreme velocities at the start/end
buffer = 0
