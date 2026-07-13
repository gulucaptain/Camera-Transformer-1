import random
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.add_subplot(projection='3d')
        self.plotly_data = None  # plotly data traces
        self.ax.set_aspect("auto")
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

    def extrinsic2pyramid(self, extrinsic, color_map='red', hw_ratio=9 / 16, base_xval=1, zval=3):
        vertex_std = np.array([[0, 0, 0, 1],
                               [base_xval, -base_xval * hw_ratio, zval, 1],
                               [base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, base_xval * hw_ratio, zval, 1],
                               [-base_xval, -base_xval * hw_ratio, zval, 1]])
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [[vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                  [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                  [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                   vertex_transformed[4, :-1]]]

        color = color_map if isinstance(color_map, str) else plt.cm.rainbow(color_map)

        self.ax.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35))

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.ax, orientation='vertical',
                          label='Frame Indexes')

    def show(self):
        plt.title('Camera Trajectory')
        plt.show()

    def save(self, filename, dpi=300, bbox_inches='tight', title="", **kwargs):
        """
        保存当前图像到文件
        
        参数:
            filename (str): 保存的文件名，包含路径和扩展名
            dpi (int): 图像分辨率，默认为300
            bbox_inches (str): 边界框设置，默认为'tight'以去除多余空白
            **kwargs: 传递给matplotlib的savefig方法的其他参数
        """
        plt.title(title)
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches,** kwargs)
        print(f"图像已保存至: {filename}")


def get_c2w(w2cs):
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ np.linalg.inv(w2c) for w2c in w2cs[1:]]
    camera_positions = np.asarray([c2w[:3, 3] for c2w in ret_poses])  # [n_frame, 3]
    position_distances = [camera_positions[i] - camera_positions[i - 1] for i in range(1, len(camera_positions))]
    xyz_max = np.max(camera_positions, axis=0)
    xyz_min = np.min(camera_positions, axis=0)
    xyz_ranges = xyz_max - xyz_min  # [3, ]
    max_range = np.max(xyz_ranges)
    expected_xyz_ranges = 1
    scale_ratio = expected_xyz_ranges / max_range
    scaled_position_distances = [dis * scale_ratio for dis in position_distances]  # [n_frame - 1]
    scaled_camera_positions = [camera_positions[0], ]
    scaled_camera_positions.extend([camera_positions[0] + np.sum(np.asarray(scaled_position_distances[:i]), axis=0)
                                    for i in range(1, len(camera_positions))])
    ret_poses = [np.concatenate(
        (np.concatenate((ori_pose[:3, :3], cam_position[:, None]), axis=1), np.asarray([0, 0, 0, 1])[None]), axis=0)
                 for ori_pose, cam_position in zip(ret_poses, scaled_camera_positions)]
    transform_matrix = np.asarray([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).reshape(4, 4)
    ret_poses = [transform_matrix @ x for x in ret_poses]
    return np.array(ret_poses, dtype=np.float32)


def visualize_trajectory(trajectory_file, save_pth, title):
    file_name = trajectory_file.split("/")[-1].split(".")[0]
    with open(trajectory_file, 'r') as f:
        poses = f.readlines()
    w2cs = [np.asarray([float(p) for p in pose.strip().split(' ')[7:]]).reshape(3, 4) for pose in poses[:13]]
    num_frames = len(w2cs)
    last_row = np.zeros((1, 4))
    last_row[0, -1] = 1.0
    w2cs = [np.concatenate((w2c, last_row), axis=0) for w2c in w2cs]
    c2ws = get_c2w(w2cs)
    visualizer = CameraPoseVisualizer([-1.2, 1.2], [-1.2, 1.2], [-1.2, 1.2])
    for frame_idx, c2w in enumerate(c2ws):
        visualizer.extrinsic2pyramid(c2w, frame_idx / num_frames, hw_ratio=9 / 16, base_xval=0.02, zval=0.1)
    visualizer.colorbar(num_frames)
    visualizer.save(save_pth, title=title)


if __name__=="__main__":
    txt_pth = "/Users/gulucaptain/Documents/code/pose_files/in_extrinsic_0ed8b86b87a30d38.txt"
    vis_traj = visualize_trajectory(txt_pth)
