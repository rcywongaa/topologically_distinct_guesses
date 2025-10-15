import os
import pickle
from types import SimpleNamespace

'''
This cannot be placed in `planning_setting.py` because then, we won't be able to retrieve the path to `randomized_scene.yaml`.
'''

dir_path = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(dir_path, 'randomized_scene.yaml')
# with open(file_path) as file:
#     data = yaml.safe_load(file)
file_path = os.path.join(dir_path, 'randomized_scene.pkl')
with open(file_path, 'rb') as file:
    data = pickle.load(file)
scene = SimpleNamespace(**data)
