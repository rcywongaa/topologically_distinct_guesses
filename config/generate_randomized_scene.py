import numpy as np
from math import pi
import random
import inspect
import pickle

'''
This script makes me sick
'''

# min_radius = 0.1
# max_radius = 0.4
radius = 0.15
sphere_obstacle_specs = []
# num_obstacles = random.randint(2, 10)
num_obstacles = 5
num_failures = 0
max_num_failures = 100
while len(sphere_obstacle_specs) < num_obstacles:
    position = (random.uniform(-0.9 + radius, 0.9 - radius), random.uniform(-0.4 +radius, 0.4 - radius), 0.0)
    for obs in sphere_obstacle_specs:
        if np.linalg.norm(np.array(obs[0]) - np.array(position)) < 2 * radius + 0.1:
            num_failures += 1
            if num_failures > max_num_failures:
                sphere_obstacle_specs.clear()
            break
    else:
        sphere_obstacle_specs.append((position, radius))

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
initial_theta = random.uniform(-pi, pi)
final_x_b = np.array([1.0, 0.1, 0.0])
final_theta = random.uniform(-pi, pi)

num_paths = 20
max_duration_s = 10

T = 100
dt = 0.2

# This gives some time for robot to orient itself before moving eef.
# Avoids requiring extreme velocities at the start/end
buffer = 0


def serialize(obj):
    """Recursively convert unsupported types to YAML-friendly formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [serialize(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    else:
        return obj

data = {}
curr_vars = vars().copy()
for key, val in curr_vars.items():
    if key.startswith("__") or key.endswith("__") or callable(val) or inspect.ismodule(val):
        continue
    # yaml_data[key] = val
    data[key] = val
# with open("randomized_scene.yaml", "w") as f:
#     yaml.dump(yaml_data, f, default_flow_style=False)
with open("randomized_scene.pkl", "wb") as f:
    pickle.dump(data, f)
