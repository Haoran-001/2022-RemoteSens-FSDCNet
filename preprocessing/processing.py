import open3d as o3d
import numpy as np
import glob
import os
import pandas as pd

names = ['t', 'intensity', 'id',
             'x', 'y', 'z',
             'azimuth', 'range', 'pid']

formats = ['int64', 'uint8', 'uint8',
               'float32', 'float32', 'float32',
               'float32', 'float32', 'int32']

binType = np.dtype(dict(names=names, formats=formats))

def generate_one_pcd_file(file_path: str):
    saved_filename = file_path.split('/')[-1].rsplit('.', 1)[0] + '.pcd'
    data = np.fromfile(file_path, binType)
    points = np.vstack([data['x'], data['y'], data['z']]).T
    for i in range(3):
        points[:, i] = (points[:, i] - np.min(points[:, i])) / (np.max(points[:, i]) - np.min(points[:, i]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd = pcd.remove_statistical_outlier(nb_neighbors = 10, std_ratio = 2.0)[0]
    o3d.io.write_point_cloud('specified_path/' + saved_filename, pcd)

def off2pcd(off_file: str = '', number_of_points: int = 9999):
    if off_file.rsplit('/', 1)[1].split('.')[1]  != 'off':
        return
    mesh = o3d.io.read_triangle_mesh(off_file)
    if len(np.asarray(mesh.vertices)) == 0:
        return
    pcd = mesh.sample_points_uniformly(number_of_points = number_of_points)
    out_file_list = off_file.split('/')
    out_file_list[5] = 'ModelNet_PCD'
    out_file = '/'.join(out_file_list)
    out_file = out_file.rsplit('.', 1)[0] + '.pcd'
    if os.path.isfile(out_file):
        return
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    o3d.io.write_point_cloud(out_file, pcd)

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert = True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom = 0.3412,
                                      front = [0.4257, -0.2125, -0.8795],
                                      lookat = [2.6172, 2.0475, 1.532],
                                      up = [-0.0694, -0.9768, 0.2024])

if __name__ == '__main__':
    # convert to pcd file for ModelNet40
    off_files = glob.glob(r'modelnet40_data_path/*/*/*')
    for off_file in off_files:
        off2pcd(off_file, 9999)

    # convert to pcd file for Sydney urban object
    bin_files = glob.glob(r'sydney_data_path/*.bin')
    for bin_file in bin_files:
        generate_one_pcd_file(bin_file)

    