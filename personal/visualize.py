# -*- coding: utf-8 -*-
""" 
@Time    : 2022/7/28 10:53
@Author  : Johnson
@FileName: visualize.py
"""
import open3d as o3d
import numpy as np


ply = o3d.io.read_point_cloud('/home/dingchaofan/point2mesh/data/modified.ply', 'auto', True, True)
print(ply)
print(np.asarray(ply.points))
print(np.asarray(ply.points).shape)

# ply.paint_uniform_corlor([0,0,1])
# # 法线估计 : ALT + - 号加长减小法线
radius = 0.01  # 搜索半径
max_nn = 30  # 邻域内用于估算法线的最大点数
ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

print(np.asarray(ply.normals))
print(np.asarray(ply.normals).shape)
# o3d.visualization.draw_geometries([ply],
#                                   window_name='Open3D',
#                                   width=800,
#                                   height = 600,
#                                   point_show_normal=True)


def read_ply_open3d():
    import open3d as o3d
    import numpy as np

    ply = o3d.io.read_point_cloud('/home/dingchaofan/point2mesh/data/modified.ply', 'auto', True, True)
    print(ply)
    print(np.asarray(ply.points))
    print(np.asarray(ply.points).shape)

    # ply.paint_uniform_corlor([0,0,1])
    # # 法线估计 : ALT + - 号加长减小法线
    radius = 0.01  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    ply.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))

    print(np.asarray(ply.normals))
    print(np.asarray(ply.normals).shape)

    return np.asarray(ply.points), np.asarray(ply.normals)