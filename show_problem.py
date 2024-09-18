import numpy as np
import uuid
from time import sleep

from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.all import MeshcatVisualizer
from pydrake.geometry import (
    Rgba,
    Sphere,
    StartMeshcat,
)

from config import planning_setting
from dataformat import get_x

# eef_height = 0.1059 + 0.0615

def show_eef(meshcat, x_e, is_persist=True):
    radius = 0.01
    link_color = Rgba(0.0, 0.0, 0.0, 0.25)
    link_width = 1  # pixels

    x_e_name = "x_e"
    # x_e[2] -= eef_height
    upperarm_name = "upperarm"
    forearm_name = "forearm"
    if is_persist:
        x_e_name = str(uuid.uuid4())
        upperarm_name = str(uuid.uuid4())
        forearm_name = str(uuid.uuid4())

    meshcat.SetTransform(x_e_name, RigidTransform(x_e))
    meshcat.SetObject(x_e_name, Sphere(radius), Rgba(0, 0, 1, 0.5))


if __name__ == "__main__":
    meshcat = StartMeshcat()
    planning_setting.show_obstacles(meshcat)

    for t in np.linspace(0, 1, 1000):
        show_eef(meshcat, planning_setting.get_eef_position(t))

    while True:
        sleep(1)
