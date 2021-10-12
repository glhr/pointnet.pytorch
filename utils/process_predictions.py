import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def display_pointcloud(xyz, colors, save=None, display=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    # vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json("RenderOption_2021-10-12-14-21-41.json")
    # if save is not None: vis.capture_screen_image(f'../results/{save}.png')
    # vis.run()
    # vis.destroy_window()

    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("renderoptions.json")
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    if save is not None: vis.capture_screen_image(f'../results/{save}.png')
    if display: vis.run()
    vis.destroy_window()

def extract_workpiece(xyz, predictions, save=None):
    mask = predictions==1
    mask = np.stack((np.squeeze(mask,1),)*3, axis=-1)
    # print(xyz.shape, predictions.shape)
    xyz = np.where(mask, xyz, 0)
    # print(type(xyz),xyz.shape,xyz[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color=[0.5,0.5,0.5])
    # o3d.visualization.draw_geometries([pcd])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.get_render_option().load_from_json("renderoptions.json")
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
