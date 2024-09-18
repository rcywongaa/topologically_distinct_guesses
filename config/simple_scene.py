import numpy as np
from math import pi

sphere_obstacle_specs = [
    ((0.5, 0.0, 0.0), 0.25),
    ((-0.5, 0.0, 0.0), 0.25),
]

aabb_obstacle_specs = []

cylinder_obstacle_specs = []

link_radius = 0.0001
base_radius = 0.0001
forearm_length = 0.4
upperarm_length = 0.3
arm_base_height = 0.0
mobile_base_height = 0.0

eef_start = [-1.0, 0.0, 0.5]
eef_end = [1.0, 0.0, 0.5]
initial_x_b = np.array([-1.0, 0.1, 0.0])
initial_theta = -0.5 * pi
final_x_b = np.array([1.0, 0.1, 0.0])
final_theta = 0.5 * pi

num_paths = 4

T = 100
dt = 0.2

# This gives some time for robot to orient itself before moving eef.
# Avoids requiring extreme velocities at the start/end
buffer = 0
