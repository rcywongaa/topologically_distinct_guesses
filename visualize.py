import os
import argparse

from pydrake.geometry import Box
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet, StartMeshcat
from pydrake.visualization import ModelVisualizer
from pydrake.multibody.tree import SpatialInertia
from pydrake.multibody.plant import CoulombFriction
from pydrake.math import RigidTransform
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument('srdf_filename', nargs='?', default=None)
args = parser.parse_args()

meshcat = StartMeshcat()

def get_collision_geometries(plant, body_name):
    try:
        return plant.GetCollisionGeometriesForBody(plant.GetBodyByName(body_name))
    except RuntimeError as e:
        print(f"Could not find {body_name}")
        return

def disable_collision(plant, collision_filter_manager, allowed_collision_pair):
    declaration = CollisionFilterDeclaration()
    set1 = GeometrySet()
    set2 = GeometrySet()
    set1_geometries = get_collision_geometries(plant, allowed_collision_pair[0])
    if set1_geometries is None:
        return
    set2_geometries = get_collision_geometries(plant, allowed_collision_pair[1])
    if set2_geometries is None:
        return
    set1.Add(set1_geometries)
    set2.Add(set2_geometries)
    declaration.ExcludeBetween(set1, set2)
    collision_filter_manager.Apply(declaration)

def load_srdf_disabled_collisions(srdf_file, plant, collision_filter_manager):
    tree = ET.parse(srdf_file)
    robot = tree.getroot()
    for disable_collisions in robot.iter('disable_collisions'):
        allowed_collision_pair = [disable_collisions.get('link1'), disable_collisions.get('link2')]
        disable_collision(plant, collision_filter_manager, allowed_collision_pair)

def create_box_obstacle(plant, name, position, dimension):
    model_instance = plant.AddModelInstance(name)
    body = plant.AddRigidBody(name, model_instance, SpatialInertia.SolidBoxWithMass(1, dimension[0], dimension[1], dimension[2]))
    plant.RegisterVisualGeometry(body, RigidTransform.Identity(), Box(dimension[0], dimension[1], dimension[2]), name+"_visual", [0.2, 0.2, 0.2, 1])
    plant.RegisterCollisionGeometry(body, RigidTransform.Identity(), Box(dimension[0], dimension[1], dimension[2]), name+"_collision", CoulombFriction(1, 1))
    plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(p=position))

def visualize_model(filename, srdf_filename):
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.parser().AddModels(filename)
    plant = visualizer.parser().plant()
    create_box_obstacle(plant, "obstacle1", [0.5, -0.5, 0.0], [0.1, 0.1, 0.5])
    if srdf_filename is not None:
        load_srdf_disabled_collisions(srdf_filename, plant, visualizer._builder.scene_graph().collision_filter_manager())
    visualizer.Run()

visualize_model(args.filename, args.srdf_filename)
