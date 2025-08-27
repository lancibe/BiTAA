import numpy as np
from plyfile import PlyData, PlyElement
import math

def rotate_ply(input_path, output_path, axis='z', degrees=90, center=(0,0,0)):
    """
    旋转PLY文件中的顶点坐标
    
    参数:
        input_path (str): 输入PLY文件路径
        output_path (str): 输出PLY文件路径
        axis (str): 旋转轴 ('x', 'y' 或 'z')
        degrees (float): 旋转角度（度数）
        center (tuple): 旋转中心点 (x, y, z)
    """
    # 加载PLY文件
    ply_data = PlyData.read(input_path)
    vertices = ply_data['vertex'].data
    has_normal = 'nx' in vertices.dtype.names

    # 将顶点转换为numpy数组
    points = np.array([[v['x'], v['y'], v['z']] for v in vertices])
    
    # 转换为弧度
    theta = np.radians(degrees)
    
    # 构建旋转矩阵
    if axis.lower() == 'x':
        rot_matrix = np.array([
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)]
        ])
    elif axis.lower() == 'y':
        rot_matrix = np.array([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
    elif axis.lower() == 'z':
        rot_matrix = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis, must be 'x', 'y' or 'z'")

    # 应用旋转（考虑旋转中心）
    points_centered = points - np.array(center)
    rotated_points = np.dot(points_centered, rot_matrix.T)
    final_points = rotated_points + np.array(center)

    # 更新顶点数据
    for i, v in enumerate(vertices):
        v['x'], v['y'], v['z'] = final_points[i]
        # 如果存在法线，也需要旋转法线
        if has_normal:
            normal = np.array([v['nx'], v['ny'], v['nz']])
            rotated_normal = np.dot(normal, rot_matrix.T)
            v['nx'], v['ny'], v['nz'] = rotated_normal

    # 保存旋转后的PLY文件
    PlyData(ply_data).write(output_path)

# 使用示例
if __name__ == "__main__":
    input_file = "audi.a2.ply"
    output_file = "audi.a2.rotated.ply"
    final_file = "audi.a2.final.ply"
    
    # 示例：绕Z轴旋转90度，以原点为中心
    rotate_ply(input_file, output_file, axis='x', degrees=270)
    
    rotate_ply(output_file, final_file, axis='y', degrees=270)
    
    # 如果需要绕模型中心旋转，可以先计算中心点：
    # ply_data = PlyData.read(input_file)
    # vertices = ply_data['vertex'].data
    # points = np.array([[v['x'], v['y'], v['z']] for v in vertices])
    # center = np.mean(points, axis=0)
    # rotate_ply(input_file, output_file, axis='y', degrees=45, center=center)