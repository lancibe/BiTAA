import torch

def compute_3D_filter(gaussians, c2w_matrices, focal_length, image_width, image_height):
    """
    计算 3D 过滤器，并将其直接应用到 gaussians 的 scale 和 opacity。

    参数：
    - gaussians: [B, N, 14] 形状的高斯数据
    - c2w_matrices: (N_views, 4, 4) 形状的 C2W 矩阵
    - focal_length: 摄像机焦距
    - image_width, image_height: 摄像机分辨率
    """

    print("[INFO] Computing 3D filter for gaussians...")

    B, N, _ = gaussians.shape
    device = gaussians.device

    # 提取 3D 位置信息
    pos = gaussians[..., 0:3]  # [B, N, 3]
    
    # 初始化距离
    distance = torch.full((B, N, 1), float('inf'), device=device)
    valid_points = torch.zeros((B, N, 1), dtype=torch.bool, device=device)

    for c2w in c2w_matrices:
        R = c2w[:3, :3].to(device)  # 旋转矩阵
        T = c2w[:3, 3].to(device)    # 平移向量

        # 转换到相机坐标系
        pos_cam = torch.matmul(pos, R.T) + T[None, None, :]

        depth = pos_cam[..., 2:3]

        # 计算投影到屏幕空间
        x_proj = pos_cam[..., 0:1] / depth * focal_length + image_width / 2.0
        y_proj = pos_cam[..., 1:2] / depth * focal_length + image_height / 2.0

        # 过滤屏幕外的点
        in_screen = (x_proj >= 0) & (x_proj < image_width) & (y_proj >= 0) & (y_proj < image_height) & (depth > 0.2)

        # 更新最近距离
        distance[in_screen] = torch.min(distance[in_screen], depth[in_screen])
        valid_points |= in_screen

    # 确保所有点都有有效距离
    distance[~valid_points] = distance[valid_points].max()

    # 计算 3D 过滤器
    filter_3D = (distance / focal_length * (0.2 ** 0.5))

    return filter_3D

def apply_3D_filter(gaussians, filter_3D):
    """
    在后处理中应用 3D 滤波器，调整 `scale` 和 `opacity`。

    参数：
    - gaussians: [B, N, 14] 的 3D 高斯数据
    - filter_3D: [B, N, 1] 的 3D 滤波器
    """

    print("[INFO] Applying 3D filter to scaling and opacity...")

    # 提取尺度和透明度
    scale = gaussians[..., 4:7]  # [B, N, 3]
    opacity = gaussians[..., 3:4]  # [B, N, 1]

    # 让 scale 受到 filter_3D 影响
    scale_filtered = torch.sqrt(torch.square(scale) + torch.square(filter_3D))

    # 让 opacity 受到 filter_3D 影响，避免过度扩散
    det1 = torch.prod(torch.square(scale), dim=-1, keepdim=True)
    det2 = torch.prod(torch.square(scale) + torch.square(filter_3D), dim=-1, keepdim=True)
    coef = torch.sqrt(det1 / det2)
    opacity_filtered = opacity * coef

    # 更新 gaussians
    gaussians[..., 4:7] = scale_filtered
    gaussians[..., 3:4] = opacity_filtered

    print("[INFO] 3D filter applied successfully.")
    return gaussians

from plyfile import PlyData, PlyElement
import numpy as np

def save_fused_ply(gaussians, path):
    """
    经过 3D 滤波后保存 PLY。

    参数：
    - gaussians: [B, N, 14] 的 3D 高斯数据
    - path: 保存路径
    """

    print(f"[INFO] Saving filtered PLY to {path}...")

    B, N, _ = gaussians.shape

    # 提取数据
    xyz = gaussians[..., 0:3].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    opacity = gaussians[..., 3:4].detach().cpu().numpy()
    scale = gaussians[..., 4:7].detach().cpu().numpy()
    rotation = gaussians[..., 7:11].detach().cpu().numpy()
    color = gaussians[..., 11:].detach().cpu().numpy()

    # 生成 PLY 结构
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z', 'nx', 'ny', 'nz', 'opacity'] +
                  [f'scale_{i}' for i in range(scale.shape[-1])] +
                  [f'rot_{i}' for i in range(rotation.shape[-1])] +
                  ['r', 'g', 'b']]

    elements = np.empty((B * N,), dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, opacity, scale, rotation, color), axis=-1)
    elements[:] = list(map(tuple, attributes.reshape(-1, attributes.shape[-1])))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print("[INFO] Filtered PLY saved successfully.")
