import open3d as o3d
import numpy as np
import cv2

# 1. Chargement des données LIDAR / 加载 LIDAR 数据
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Loaded point cloud with {len(pcd.points)} points.")
    return pcd

# 2. Réduction de la densité du nuage de points / 点云下采样
def downsample_point_cloud(pcd, voxel_size=0.05):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

# 3. Projection du nuage de points en carte de profondeur 2D / 投影点云到 2D 深度图
def project_to_depth_map(pcd, resolution=512):
    points = np.asarray(pcd.points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_norm = ((x - x.min()) / (x.max() - x.min()) * (resolution - 1)).astype(int)
    y_norm = ((y - y.min()) / (y.max() - y.min()) * (resolution - 1)).astype(int)
    z_norm = (z - z.min()) / (z.max() - z.min())

    depth_map = np.zeros((resolution, resolution))
    depth_map[y_norm, x_norm] = z_norm
    return depth_map

# 4. Sauvegarde de la carte de profondeur / 保存深度图
def save_depth_map(depth_map, file_name):
    depth_image = (depth_map * 255).astype(np.uint8)
    cv2.imwrite(file_name, depth_image)
    print(f"Depth map saved to {file_name}")

# 5. Alignement des nuages de points / 点云配准（简单刚性对齐）
def align_point_clouds(source, target):
    transformation = np.eye(4)  # Transformation par défaut 默认单位矩阵
    source.transform(transformation)
    return source

# 6. Différenciation entre objets dynamiques et statiques / 区分动态和静态物体
def differentiate_objects(source, target, threshold=0.1):
    distances = np.asarray(source.compute_point_cloud_distance(target))
    dynamic_mask = distances > threshold

    dynamic_points = np.asarray(source.points)[dynamic_mask]
    static_points = np.asarray(source.points)[~dynamic_mask]

    dynamic_pcd = o3d.geometry.PointCloud()
    static_pcd = o3d.geometry.PointCloud()
    dynamic_pcd.points = o3d.utility.Vector3dVector(dynamic_points)
    static_pcd.points = o3d.utility.Vector3dVector(static_points)

    # Appliquer des couleurs / 应用颜色
    dynamic_pcd.paint_uniform_color([1, 0, 0])  # Rouge / 红色
    static_pcd.paint_uniform_color([0, 1, 0])   # Vert / 绿色

    return dynamic_pcd, static_pcd

# 7. Visualisation des nuages de points / 点云可视化
def visualize_point_clouds(*pcds):
    o3d.visualization.draw_geometries(pcds)

# Fonction principale / 主函数
if __name__ == "__main__":
    # Chemin des fichiers / 文件路径
    file_path1 = "D:/LidarDonnes/donnes4-sequence_1.pcd"
    file_path2 = "D:/LidarDonnes/donnes4-sequence_50.pcd"

    # 1. Chargement du nuage de points / 加载点云
    pcd1 = load_point_cloud(file_path1)
    pcd2 = load_point_cloud(file_path2)

    # 2. Réduction de la densité / 下采样点云
    pcd1 = downsample_point_cloud(pcd1, voxel_size=0.05)
    pcd2 = downsample_point_cloud(pcd2, voxel_size=0.05)

    # 3. Alignement des nuages de points / 对齐点云
    aligned_pcd1 = align_point_clouds(pcd1, pcd2)

    # 4. Génération de la carte de profondeur / 生成深度图
    depth_map = project_to_depth_map(aligned_pcd1, resolution=512)
    save_depth_map(depth_map, "depth_image.png")

    # 5. Différenciation entre objets dynamiques et statiques / 区分动态与静态物体
    dynamic_pcd, static_pcd = differentiate_objects(aligned_pcd1, pcd2, threshold=0.1)
    print(f"Dynamic points: {len(dynamic_pcd.points)}, Static points: {len(static_pcd.points)}")

    # 6. Visualisation des résultats / 可视化结果
    visualize_point_clouds(static_pcd, dynamic_pcd)