import numpy as np
from math import pi

chair_x = 0.0
chair_y = -0.35
chair_height = 0.55
chair_seat_thickness = 0.05
chair_seat_radius = 0.15
chair_leg_radius = 0.025
chair_leg_length = chair_height
table_x = 0.0
table_y = 0.0
table_height = 0.9
table_thickness = 0.05
table_width = 1.5
table_depth = 0.2
table_overhang = 0.00
table_leg_length = table_height - table_thickness
table_leg_depth = table_depth / 2

sphere_obstacle_specs = []

aabb_obstacle_specs = [
    (
        (table_x, table_y, table_height - table_thickness / 2),
        (table_width, table_depth, table_thickness),
    ),  # table top
    (
        (
            -(table_width / 2 - table_overhang - table_thickness / 2),
            table_y,
            table_leg_length / 2,
        ),
        (table_thickness, table_leg_depth, table_leg_length),
    ),  # left leg
    (
        (
            table_width / 2 - table_overhang - table_thickness / 2,
            table_y,
            table_leg_length / 2,
        ),
        (table_thickness, table_leg_depth, table_leg_length),
    ),  # right leg
]

cylinder_obstacle_specs = [
    (
        (chair_x, chair_y, chair_height - chair_seat_thickness / 2),
        (chair_seat_radius, chair_seat_thickness),
    ),  # chair seat
    (
        (chair_x, chair_y, chair_leg_length / 2),
        (chair_leg_radius, chair_leg_length),
    ),  # chair leg
]

link_radius = 0.05
# base_radius = 0.29  # 0.573/2
base_radius = 0.1705

forearm_length = 0.3143  # 0.2084 + 0.1059
# forearm_length = 0.2084 + 0.1059 + 0.1059 + 0.0615
upperarm_length = 0.41
mobile_base_height = 0.359
arm_base_height = 0.6438  # mobile_base_height + 0.1284 + 0.1564
eef_height = table_height + 0.1059 + 0.0615 + 0.02
# eef_height = 1.05
eef_start = [-1.0, 0.0, eef_height]
eef_end = [1.0, 0.0, eef_height]
initial_theta = 0.01
initial_x_b = np.array([-1.4, 0.05, arm_base_height])
final_theta = 0.01
final_x_b = np.array([1.4, 0.05, arm_base_height])

num_paths = 3

T = 200
dt = 0.2

# This gives some time for robot to orient itself before moving eef.
# Avoids requiring extreme velocities at the start/end
buffer = 10
