import numpy as np
import open3d as o3d
from scipy.spatial import KDTree

class GeometryEvaluator:
    def __init__(self, original_ply, generated_ply):
        """
        初始化类，读取 PLY 文件并提取点云数据
        """
        self.original_points = self.load_ply(original_ply)
        self.generated_points = self.load_ply(generated_ply)

    def load_ply(self, file_path):
        """
        使用 Open3D 读取 PLY 文件中的点云数据
        """
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            raise ValueError(f"文件 {file_path} 可能为空或格式错误")
        return np.asarray(pcd.points)  # 返回点云的坐标数组 (N, 3)

    def chamfer_distance(self):
        """
        计算 Chamfer Distance (CD)
        """
        tree_orig = KDTree(self.original_points)
        tree_gen = KDTree(self.generated_points)

        # 计算每个点到最近邻点的距离
        dist1, _ = tree_orig.query(self.generated_points)
        dist2, _ = tree_gen.query(self.original_points)

        # Chamfer 距离是两个方向的平均
        chamfer_dist = np.mean(dist1) + np.mean(dist2)
        return chamfer_dist

    def hausdorff_distance(self):
        """
        计算 Hausdorff Distance (HD)
        """
        tree_orig = KDTree(self.original_points)
        tree_gen = KDTree(self.generated_points)

        # 计算从每个点到对方点云的最大最小距离
        dist1, _ = tree_orig.query(self.generated_points)
        dist2, _ = tree_gen.query(self.original_points)

        hausdorff_dist = max(np.max(dist1), np.max(dist2))
        return hausdorff_dist

    def mass_centroid_alignment(self):
        """
        计算 Mass Centroid Alignment (MCA)
        """
        centroid_orig = np.mean(self.original_points, axis=0)
        centroid_gen = np.mean(self.generated_points, axis=0)

        mca_distance = np.linalg.norm(centroid_orig - centroid_gen)
        return mca_distance

# 示例用法
if __name__ == "__main__":
    original_ply = "audi.a2.final.ply"
    # generated_ply = "workspace/0224/0_attack.ply"
    generated_ply = "workspace/0224/0.ply"

    evaluator = GeometryEvaluator(original_ply, generated_ply)

    cd = evaluator.chamfer_distance()
    hd = evaluator.hausdorff_distance()
    mca = evaluator.mass_centroid_alignment()

    print(f"Chamfer Distance: {cd:.6f}")
    print(f"Hausdorff Distance: {hd:.6f}")
    print(f"Mass Centroid Alignment: {mca:.6f}")
