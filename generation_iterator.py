import os
import trimesh
import numpy as np
from tqdm import tqdm


def gen_iterator(out_path, dataset, gen_p):

    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)


    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    for i, data in tqdm(enumerate(loader)):


        path = os.path.normpath(data['path'][0])
        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])


        if os.path.exists(export_path):
            print('Path exists - skip! {}'.format(export_path))
            continue
        else:
            os.makedirs(export_path)

        for num_steps in [7]:
            point_cloud, duration = gen.generate_point_cloud(data, num_steps)
            np.savez( export_path + 'dense_point_cloud_{}'.format(num_steps), point_cloud = point_cloud, duration = duration)
            print('num_steps', num_steps, 'duration', duration)
            trimesh.Trimesh(vertices = point_cloud, faces = []).export( export_path + 'dense_point_cloud_{}.off'.format(num_steps))
