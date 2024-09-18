import argparse            
import pathlib
from trimesh import load_mesh
from trimesh.exchange.obj import export_obj

def convert(root, extension):
    for mesh_file in pathlib.Path(root).glob(f'*/**/*.{extension}'):
        print(f"Converting {mesh_file}")
        mesh = load_mesh(mesh_file)
        obj, data = export_obj(
                    mesh, return_texture=True, mtl_name=mesh_file.with_suffix('.mtl'))
        obj_file = mesh_file.with_suffix('.obj')
        with open(obj_file, "w") as f:
            f.write(obj)
            print(f"Wrote {obj_file}")
        # save the MTL and images                                
        for k, v in data.items():
            with open(k, 'wb') as f:
                f.write(v)
                print(f"Wrote {k}")

parser = argparse.ArgumentParser("Convert meshes to obj")
parser.add_argument("root", type=str, help="The root directory to glob for stl files")
parser.add_argument("--extension", type=str, default="stl", help="The type of mesh to convert. You may specify dae. Other extensions may work as well.")

args = parser.parse_args()

convert(args.root, extension=args.extension)
