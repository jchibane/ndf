import trimesh

def from_grid(grid_points):
    points = grid_points.copy()
    points[:, 0], points[:, 2] = grid_points[:, 2], grid_points[:, 0]

    points2 = points.copy()
    points2[:,1], points2[:,2] = points[:,2] , points[:,1]
    return 0.5 * points2

if __name__ == "__main__":
    file_name  = '/BS/chiban4/work/IDFF-Net/paper_results/1mio_iGarments_dist-0.02_0.48_0.5_sigmas-0.08_0.02_0.01_v256_mSVR_4.off'
    save_path = '/BS/chiban5/work/garments/paper/1_mio_index4.off'
    mesh = trimesh.load_mesh(file_name)

    new_verts = from_grid(mesh.vertices)
    trimesh.Trimesh(vertices = new_verts, faces = mesh.faces).export(save_path)

    file_name  = '/BS/chiban4/work/IDFF-Net/paper_results/1mio_iGarments_dist-0.02_0.48_0.5_sigmas-0.08_0.02_0.01_v256_mSVR_12.off'
    save_path = '/BS/chiban5/work/garments/paper/1_mio_index12.off'
    mesh = trimesh.load_mesh(file_name)

    new_verts = from_grid(mesh.vertices)
    trimesh.Trimesh(vertices = new_verts, faces = mesh.faces).export(save_path)