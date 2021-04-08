import os
import numpy as np
import trimesh
import argparse

def get_dirs_paths(d):
    paths = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    dirs = [ o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
    return sorted(dirs), sorted(paths)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type = str)
    parser.add_argument("--output_folder", type = str)
    args = parser.parse_args()

    lists = ['TShirtNoCoat.obj', 'ShortPants.obj', 'Pants.obj', 'ShirtNoCoat.obj', 'LongCoat.obj']

    all_dirs, all_paths = get_dirs_paths(args.input_folder)
    for index in range(len(all_paths)):
        path = all_paths[index]
        for file in os.listdir(path):
            if file in lists:
                class_name = file.replace('.obj', '')
                mesh_path = os.path.join(path, file)
                out_dir = os.path.join(args.output_folder, class_name + '_' + all_dirs[index])
                if not os.path.isdir(out_dir):
                    os.makedirs(out_dir)

                out_file = os.path.join(out_dir, 'mesh.off')

                mesh = trimesh.load(mesh_path)

                new_verts = mesh.vertices - np.mean(mesh.vertices, axis = 0)
                new_verts_sc = new_verts / 0.9748783846
                new_verts_sc = new_verts_sc * 0.5
                new_mesh = trimesh.Trimesh(vertices = new_verts_sc, faces = mesh.faces)
                new_mesh.export(out_file)
                print("Processed {} {}".format(all_dirs[index], class_name))
