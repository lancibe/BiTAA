import torch
import numpy as np
import tqdm
import os
import math
from PIL import Image
from plyfile import PlyData

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def orbit_camera(elevation, azimuth, radius=4.0):
    """生成相机视图矩阵（世界到相机）"""
    azimuth = math.radians(azimuth)
    elevation = math.radians(elevation)
    
    # 计算相机位置（球坐标系）
    x = radius * math.cos(azimuth) * math.cos(elevation)
    y = radius * math.sin(elevation)
    z = radius * math.sin(azimuth) * math.cos(elevation)
    pos = np.array([x, y, z])
    
    # 构建视图矩阵
    forward = -pos / np.linalg.norm(pos)
    right = np.cross(forward, np.array([0, 1, 0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    
    view_matrix = np.eye(4)
    view_matrix[:3, 0] = right
    view_matrix[:3, 1] = up
    view_matrix[:3, 2] = forward
    view_matrix[:3, 3] = pos
    return np.linalg.inv(view_matrix)  # 返回相机到世界的变换矩阵

def load_ply(path, device):
    """安全的PLY加载实现"""
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # 基础数据转换（直接创建CUDA张量）
    positions = torch.tensor(np.vstack([vertex['x'], vertex['y'], vertex['z']]).T, 
                    dtype=torch.float32, device=device)
    colors = torch.tensor(np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T / 255.0,
                    dtype=torch.float32, device=device)
    
    # 生成验证过的伪参数
    num_points = positions.shape[0]
    return {
        'positions': positions.clone().detach(),
        'colors': colors.clone().detach(),
        'opacities': torch.sigmoid(torch.ones(num_points, 1, device=device) * 0.8),
        'scales': torch.exp(torch.full((num_points, 3), math.log(0.03), device=device)),
        'rotations': generate_safe_rotations(positions, device)
    }


def load_legacy_ply(path, device):
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # 基础属性
    positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T / 255.0
    normals = np.vstack([vertex['nx'], vertex['ny'], vertex['nz']]).T
    
    # 生成伪高斯参数（关键修改：限制参数范围）
    num_points = positions.shape[0]
    return {
        'positions': torch.tensor(positions, dtype=torch.float32, device=device),
        'colors': torch.tensor(colors, dtype=torch.float32, device=device),
        'opacities': torch.sigmoid(torch.full((num_points, 1), 1.5, device=device)),  # sigmoid(1.5)=0.82
        'scales': torch.exp(torch.full((num_points, 3), math.log(0.03), device=device)),  # 0.03米
        'rotations': generate_safe_rotations(normals, device)
    }

# 修正旋转生成函数
def generate_safe_rotations(positions, device):
    """生成安全的各向同性旋转"""
    num_points = positions.shape[0]
    rotations = torch.zeros((num_points, 4), device=device)
    rotations[:, 0] = 1.0  # 单位四元数 [w=1, x=0, y=0, z=0]
    return rotations  # 简化处理，禁用法线对齐

def build_rotation(q):
    """安全构建旋转矩阵（带归一化处理）"""
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-6)  # 增加数值稳定性
    w, x, y, z = q.unbind(-1)
    
    return torch.stack([
        1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)
    ], dim=-1).view(*q.shape[:-1], 3, 3)

def build_scaling_rotation(scales, rotations):
    """验证协方差矩阵计算"""
    # 确保输入维度正确
    assert scales.shape == (scales.size(0), 3), f"Scales shape error: {scales.shape}"
    assert rotations.shape == (rotations.size(0), 4), f"Rotations shape error: {rotations.shape}"
    
    R = build_rotation(rotations)
    S = scales.unsqueeze(-1) * torch.eye(3, device=scales.device)
    return torch.bmm(R, torch.bmm(S, R.transpose(-1, -2)))

def strip_symmetric(cov):
    """
    提取协方差矩阵的对称部分
    """
    return cov[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]]

def get_projection_matrix(fov, aspect, near, far, device):
    """修正投影矩阵生成（显式设备指定）"""
    tan_half_fov = math.tan(math.radians(fov) / 2)
    proj = torch.zeros((4, 4), device=device)
    proj[0, 0] = 1 / (aspect * tan_half_fov)
    proj[1, 1] = 1 / tan_half_fov
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1
    return proj.unsqueeze(0).transpose(1, 2)

def main():
    # 配置参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "./rendered_views"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载PLY文件
    # data = load_ply("audi.a2.final.ply")
    data = load_ply("audi.a2.final.ply", device=device)
    print(f"数据验证: positions={data['positions'].shape}, colors={data['colors'].shape}, "
        f"scales={data['scales'].shape}, rotations={data['rotations'].shape}")
    
    # 计算协方差矩阵
    with torch.no_grad():
        covariance = build_scaling_rotation(data['scales'], data['rotations'])
        print(f"协方差矩阵形状: {covariance.shape}")

    gaussians_pos = torch.tensor(data['positions'], dtype=torch.float32, device=device)
    gaussians_rgb = torch.tensor(data['colors'], dtype=torch.float32, device=device)
    opacities = torch.tensor(data['opacities'], dtype=torch.float32, device=device)
    scales = torch.tensor(data['scales'], dtype=torch.float32, device=device)
    rotations = torch.tensor(data['rotations'], dtype=torch.float32, device=device)
    
    # from utils.graphics_utils import strip_symmetric, build_scaling_rotation
    covariance = build_scaling_rotation(scales, rotations)
    cov3D_precomp = [strip_symmetric(covariance)]    
    
    # 渲染参数
    image_w, image_h = 512, 512
    fov = 90
    proj_matrix = get_projection_matrix(fov, image_w/image_h, 0.1, 100, device)
    
    # 渲染循环
    for idx, azi in enumerate(tqdm.tqdm(np.linspace(0, 360, 12, endpoint=False))):
        # 计算相机位姿（保持4x4矩阵）
        cam_pose = orbit_camera(30, azi, radius=2.0)  # 减小观察距离
        cam_pose = torch.from_numpy(cam_pose).unsqueeze(0).to(device)
        
        # 坐标系调整（关键修改：仅调整旋转部分）
        cam_pose[:, :3, 1:3] *= -1
        
        # 视图矩阵计算
        view_matrix = torch.inverse(cam_pose).transpose(1, 2).float()
        view_proj_matrix = view_matrix @ proj_matrix
        
        # 创建光栅化设置
        raster_settings = GaussianRasterizationSettings(
            image_height=image_h,
            image_width=image_w,
            tanfovx=math.tan(math.radians(fov/2)),
            tanfovy=math.tan(math.radians(fov/2)),
            bg=torch.tensor([1,1,1], dtype=torch.float32, device=device),  # 背景设为白色
            scale_modifier=1.0,
            viewmatrix=view_matrix.squeeze(0),
            projmatrix=view_proj_matrix.squeeze(0),
            sh_degree=0,  # 如果使用预计算颜色则为0
            campos=cam_pose[:, :3, 3],
            prefiltered=False,
            debug=False
        )
        
        # 创建光栅化器
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        try:
            # 需要确保加载了所有高斯参数（关键修改！）
            rendered_image, radii = rasterizer(
                means3D=gaussians_pos,           # [N,3]
                means2D=torch.zeros_like(gaussians_pos),  # 占位符（实际未使用）
                shs=None,                         # 使用预计算颜色时设为None
                colors_precomp=gaussians_rgb,     # [N,3]
                opacities=opacities,              # [N,1] 需要从PLY加载
                scales=scales,                    # [N,3] 需要从PLY加载
                rotations=rotations,              # [N,4] 需要从PLY加载
                cov3D_precomp=cov3D_precomp       # [N,6] 协方差矩阵（可选）
            )
            
            # 调整输出格式
            rendered_image = rendered_image.clamp(0,1)[:3]  # 取前三个通道（RGB）
            
            # 保存图像
            img = (rendered_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(img).save(f"{output_dir}/view_{idx:03d}.png")
        
        except Exception as e:
            print(f"渲染失败于视角 {idx}: {str(e)}")
            continue

if __name__ == "__main__":
    main()