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
