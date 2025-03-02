import open3d as o3d
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # 示例：使用随机森林
from sklearn.preprocessing import StandardScaler     # 用于特征归一化的示例

# ------------------ 1. 加载 LIDAR 数据 ------------------
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    return pcd

# ------------------ 2. 点云下采样 ------------------
def downsample_point_cloud(pcd, voxel_size=0.05):
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Downsampled to {len(downsampled_pcd.points)} points.")
    return downsampled_pcd

# ------------------ 3. 投影点云到 2D 深度图 ------------------
def project_to_depth_map(pcd, resolution=512):
    points = np.asarray(pcd.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_norm = ((x - x.min()) / (x.max() - x.min()) * (resolution - 1)).astype(int)
    y_norm = ((y - y.min()) / (y.max() - y.min()) * (resolution - 1)).astype(int)
    z_norm = (z - z.min()) / (z.max() - z.min())

    depth_map = np.zeros((resolution, resolution))
    depth_map[y_norm, x_norm] = z_norm
    return depth_map

# ------------------ 4. 以 OpenCV 方式保存彩色深度图 ------------------
def save_depth_map_colored(depth_map, file_name, colormap=cv2.COLORMAP_JET):
    depth_image = (depth_map * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_image, colormap)
    cv2.imwrite(file_name, colored_depth)
    print(f"Colored depth map saved to {file_name}")

# ------------------ 5. 区分动态和静态物体 ------------------
def differentiate_objects(source, target, threshold=0.1):
    distances = np.asarray(source.compute_point_cloud_distance(target))
    dynamic_mask = distances > threshold

    dynamic_points = np.asarray(source.points)[dynamic_mask]
    static_points = np.asarray(source.points)[~dynamic_mask]

    dynamic_pcd = o3d.geometry.PointCloud()
    static_pcd = o3d.geometry.PointCloud()
    dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points)
    static_pcd.points = o3d.utility.Vector3dVector(static_points)

    # 应用颜色
    dynamic_pcd.paint_uniform_color([1, 0, 0])  # 红色
    static_pcd.paint_uniform_color([0, 1, 0])   # 绿色

    return dynamic_pcd, static_pcd

# ------------------ 6. 点云可视化 ------------------
def visualize_point_clouds(*pcds):
    o3d.visualization.draw_geometries(pcds)

# ------------------ 7. DBSCAN 聚类 ------------------
def dbscan(pcd, eps=0.45, min_points=7, print_progress=False, debug=False):
    verbosityLevel = o3d.utility.VerbosityLevel.Warning
    if debug:
        verbosityLevel = o3d.utility.VerbosityLevel.Debug

    with o3d.utility.VerbosityContextManager(verbosityLevel):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")

    # 生成颜色
    cmap = plt.get_cmap("tab20")
    colors = cmap(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = [0, 0, 0, 1]  # 噪声点设为黑色

    # 应用颜色
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels

# ------------------ 8. 提取聚类特征 ------------------
def extract_features_from_cluster(cluster):
    """
    从聚类点云提取特征：
    1. 点数
    2. 空间包围盒尺寸(长宽高)
    3. 平均Z值
    """
    pts = np.asarray(cluster.points)
    if len(pts) == 0:
        return np.array([0, 0, 0, 0, 0])  # 空聚类返回空特征

    num_points = len(pts)
    bbox = cluster.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # (dx, dy, dz)
    mean_z = pts[:, 2].mean()

    return np.array([num_points, extent[0], extent[1], extent[2], mean_z], dtype=float)

if __name__ == "__main__":
    # 文件路径
    file_path1 = "D:/LidarDonnes/donnes4-sequence_1.pcd"
    file_path2 = "D:/LidarDonnes/donnes4-sequence_50.pcd"

    # 1. 加载点云
    pcd1 = load_point_cloud(file_path1)
    pcd2 = load_point_cloud(file_path2)

    # 2. 下采样
    pcd1 = downsample_point_cloud(pcd1, voxel_size=0.05)
    pcd2 = downsample_point_cloud(pcd2, voxel_size=0.05)

    # 3. 计算深度图
    depth_map = project_to_depth_map(pcd1, resolution=512)
    save_depth_map_colored(depth_map, "depth_image.png")

    # 4. 区分动态和静态物体
    dynamic_pcd, static_pcd = differentiate_objects(pcd1, pcd2, threshold=0.1)
    print(f"Dynamic points: {len(dynamic_pcd.points)}, Static points: {len(static_pcd.points)}")

    # 5. 可视化
    visualize_point_clouds(static_pcd, dynamic_pcd)

    # 6. DBSCAN 聚类动态物体
    clustered_pcd, labels = dbscan(dynamic_pcd, eps=0.45, min_points=20)
    print(f"Clusters found: {labels.max() + 1}")

    # 7. 可视化最终结果
    visualize_point_clouds(clustered_pcd, static_pcd)
