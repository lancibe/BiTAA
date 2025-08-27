import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from torch.distributions  import MultivariateNormal

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


class FixedViewTextureProjector:
    def __init__(self):
        self.texture_size = 2048
        self.device = 'cuda'
        
        # 定义固定视角参数（正交投影）
        self.views = {
            'front': {'axis': 'z', 'sign': 1, 'pos': [0,0,1]},
            'back':  {'axis': 'z', 'sign': -1, 'pos': [0,0,-1]},
            'left':  {'axis': 'x', 'sign': -1, 'pos': [-1,0,0]},
            'right': {'axis': 'x', 'sign': 1, 'pos': [1,0,0]},
            'top':   {'axis': 'y', 'sign': 1, 'pos': [0,1,0]}
        }
        
        # 十字纹理布局参数（3x3网格）
        self.view_regions = {
            'front': (1, 2),
            'back':  (1, 0),
            'left':  (0, 1),
            'right': (2, 1),
            'top':   (1, 1)
        }

        # self.opt = opt
        # intrinsics
        self.fovy = 90.0
        self.zfar, self.znear = 20.0, 0.5
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.zfar + self.znear) / (self.zfar - self.znear)
        self.proj_matrix[3, 2] = - (self.zfar * self.znear) / (self.zfar - self.znear)
        self.proj_matrix[2, 3] = 1
    
    def create_cross_texture(self, view_textures):
        """创建标准十字纹理布局"""
        tile_size = self.texture_size // 3
        cross_img = Image.new('RGB', (self.texture_size, self.texture_size))
        
        positions = {
            'front': (tile_size, 2*tile_size),
            'back':  (tile_size, 0),
            'left':  (0, tile_size),
            'right': (2*tile_size, tile_size),
            'top':   (tile_size, tile_size)
        }
        
        for view in ['front', 'back', 'left', 'right', 'top']:
            img = view_textures[view].resize((tile_size, tile_size))
            cross_img.paste(img, positions[view])
        
        return cross_img

    def _get_dominant_view(self, positions):
        """确定每个点的主导投影视图"""
        # 计算各坐标轴绝对值的最大方向
        abs_pos = torch.abs(positions)
        max_val, max_idx = torch.max(abs_pos, dim=1)
        
        # 创建视图选择掩码
        view_mask = torch.zeros(len(positions), dtype=torch.long)
        
        # X轴主导
        x_mask = (max_idx == 0)
        view_mask[x_mask & (positions[:,0] > 0)] = 3  # right
        view_mask[x_mask & (positions[:,0] <= 0)] = 2  # left
        
        # Y轴主导
        y_mask = (max_idx == 1)
        view_mask[y_mask] = 4  # top
        
        # Z轴主导
        z_mask = (max_idx == 2)
        view_mask[z_mask & (positions[:,2] > 0)] = 0  # front
        view_mask[z_mask & (positions[:,2] <= 0)] = 1  # back
        
        return view_mask

    def project(self, gaussian_params, cross_texture):
        """
        核心投影方法
        参数:
            gaussian_params: (N,14) 张量
            cross_texture: PIL Image对象
        返回:
            colors: (N,3) 映射后的颜色
        """
        # 转换纹理为张量
        tex_tensor = torch.tensor(np.array(cross_texture), 
                                device=self.device).float() / 255.0
        
        # 提取位置信息（假设已归一化到[-1,1]）
        positions = gaussian_params[:, 0:3]
        
        # 步骤1：确定主导视图
        view_mask = self._get_dominant_view(positions)
        
        # 步骤2：计算UV坐标
        uv = torch.zeros((len(positions), 2), device=self.device)
        tile_size = self.texture_size // 3
        
        for view_idx, view_name in enumerate(['front', 'back', 'left', 'right', 'top']):
            mask = view_mask == view_idx
            if not mask.any():
                continue
                
            view_data = self.views[view_name]
            pos_subset = positions[mask]
            
            # 根据视图类型计算投影坐标
            if view_name in ['front', 'back']:
                # 投影到XY平面，使用X,Y坐标
                proj_coords = pos_subset[:, [0, 1]]
            elif view_name in ['left', 'right']:
                # 投影到YZ平面，使用Y,Z坐标
                proj_coords = pos_subset[:, [1, 2]]
            elif view_name == 'top':
                # 投影到XZ平面，使用X,Z坐标
                proj_coords = pos_subset[:, [0, 2]]
            
            # 归一化到[0,1]范围
            proj_coords = (proj_coords + 1) / 2  # [-1,1] => [0,1]
            
            # 调整到对应纹理区域
            grid_x, grid_y = self.view_regions[view_name]
            uv[mask, 0] = (proj_coords[:,0] + grid_x) / 3
            uv[mask, 1] = (proj_coords[:,1] + grid_y) / 3
        
        # 步骤3：采样纹理颜色
        colors = F.grid_sample(
            tex_tensor.permute(2,0,1).unsqueeze(0),  # (1,C,H,W)
            uv.view(1,-1,1,2),                       # (1,N,1,2)
            align_corners=False,
            mode='bilinear'
        ).squeeze().T
        
        return colors

    def apply_to_gaussians(self, gaussian_params, cross_texture):
        """直接应用到高斯参数的便捷方法"""
        with torch.no_grad():
            new_colors = self.project(gaussian_params, cross_texture)
        updated_params = gaussian_params.clone()
        updated_params[:, 11:14] = new_colors  # 替换颜色参数
        return updated_params
    
    # def extract_view_textures(self, gaussian_params):
    #     """
    #     从3D高斯参数中提取五个面的纹理信息
    #     参数:
    #         gaussian_params: (N,14) 张量
    #     返回:
    #         view_textures: 包含五个视角纹理信息的字典
    #     """
    #     positions = gaussian_params[:, 0:3]
    #     colors = gaussian_params[:, 11:14]
        
    #     view_textures = {}
    #     tile_size = self.texture_size // 3
        
    #     for view_name in ['front', 'back', 'left', 'right', 'top']:
    #         view_data = self.views[view_name]
    #         view_mask = self._get_dominant_view(positions) == list(self.views.keys()).index(view_name)
    #         if not view_mask.any():
    #             continue
            
    #         pos_subset = positions[view_mask]
    #         color_subset = colors[view_mask]
            
    #         if view_name in ['front', 'back']:
    #             proj_coords = pos_subset[:, [0, 1]]
    #         elif view_name in ['left', 'right']:
    #             proj_coords = pos_subset[:, [1, 2]]
    #         elif view_name == 'top':
    #             proj_coords = pos_subset[:, [0, 2]]
            
    #         # proj_coords = (proj_coords + 1) / 2  # [-1,1] => [0,1]
    #         proj_coords = (proj_coords + 1) / 2 * 0.8 + 0.1  # [-1,1] => [0.1,0.9]
    #         proj_coords = (proj_coords * (tile_size - 1)).long()
            
    #         # 创建白色背景的纹理图像
    #         texture = torch.ones((tile_size, tile_size, 3), device=self.device)
    #         texture[proj_coords[:, 1], proj_coords[:, 0]] = color_subset
                        
    #         # 测试：使用插值方法生成连续的纹理
    #         texture = F.interpolate(texture.permute(2, 0, 1).unsqueeze(0), size=(tile_size, tile_size), mode='bilinear', align_corners=False).squeeze().permute(1, 2, 0)


    #         view_textures[view_name] = Image.fromarray((texture.cpu().numpy() * 255).astype(np.uint8))
        
    #     return view_textures
    
    def extract_view_textures(self, gaussian_params):
        """
        从3D高斯参数中提取五个面的纹理信息,并处理透明度遮挡效应 
        参数:
            gaussian_params: (N, 14) 张量 
        返回:
            view_textures: 包含五个视角纹理信息的字典 
        """
        # 提取相关参数 
        positions = gaussian_params[:, 0:3].to(self.device) 
        scales = gaussian_params[:, 4:7].to(self.device) 
        colors = gaussian_params[:, 11:14].to(self.device) 
        opacity = gaussian_params[:, 3:4].to(self.device) 
    
        view_textures = {}
        tile_size = self.texture_size  // 3 
    
        # 初始化光栅化设置 
        raster_settings = GaussianRasterizationSettings(
            image_height=tile_size,
            image_width=tile_size,
            tanfovx=self.tan_half_fov, 
            tanfovy=self.tan_half_fov, 
            bg=torch.zeros(3,  device=self.device),   # 背景颜色 
            scale_modifier=scale_modifier,
            viewmatrix=view_matrix,
            projmatrix=view_proj_matrix,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
    
        # 创建光栅化器实例 
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
        for view_name in ['front', 'back', 'left', 'right', 'top']:
            # 根据视图方向获取变换矩阵 
            if view_name in ['front', 'back']:
                view_matrix = self.get_front_view_matrix() 
                proj_matrix = self.get_projection_matrix() 
            elif view_name in ['left', 'right']:
                view_matrix = self.get_side_view_matrix(view_name) 
                proj_matrix = self.get_projection_matrix() 
            elif view_name == 'top':
                view_matrix = self.get_top_view_matrix() 
                proj_matrix = self.get_projection_matrix() 
    
            # 更新光栅化设置 
            raster_settings.viewmatrix  = view_matrix 
            raster_settings.projmatrix  = proj_matrix 
    
            # 调用光栅化器进行渲染 
            rendered_image, _, _, _ = rasterizer(
                means3D=positions,
                means2D=torch.zeros_like(positions,  dtype=torch.float32,  device=self.device), 
                shs=None,
                colors_precomp=colors,
                opacities=opacity,
                scales=scales,
                rotations=None,
                cov3D_precomp=None,
            )
    
            # 将渲染结果存储到纹理中 
            view_textures[view_name] = rendered_image.clamp(0,  1)
    
        return view_textures 
    


############################################################################################################
# def create_sample_image(color, size=(256, 256)):
#     """创建一个纯色的示例图像"""
#     img = Image.new('RGB', size, color)
#     return img

# def test_project_gaussians():
#     # 创建示例图像
#     view_textures = {
#         'front': create_sample_image('red'),
#         'back': create_sample_image('green'),
#         'left': create_sample_image('blue'),
#         'right': create_sample_image('yellow'),
#         'top': create_sample_image('purple')
#     }
    
#     # 实例化 FixedViewTextureProjector 类
#     projector = FixedViewTextureProjector(texture_size=2048, device='cuda:2')
    
#     # 创建十字纹理图像
#     cross_texture = projector.create_cross_texture(view_textures)
    
#     # 创建随机的3D高斯参数 (N, 14)
#     N = 1000  # 随机点的数量
#     gaussian_params = torch.randn(N, 14, device='cuda:2')
    
#     # 将位置参数归一化到 [-1, 1] 范围
#     gaussian_params[:, 0:3] = torch.tanh(gaussian_params[:, 0:3])
    
#     # 投影高斯参数到纹理上
#     new_colors = projector.project(gaussian_params, cross_texture)
    
#     # 更新高斯参数中的颜色信息
#     updated_params = projector.apply_to_gaussians(gaussian_params, cross_texture)
    
#     # 打印一些结果以进行验证
#     print("Original Gaussian Params (first 5):", gaussian_params[:5])
#     print("Updated Gaussian Params (first 5):", updated_params[:5])
    
#     # 保存生成的十字纹理图像
#     cross_texture.save('cross_texture.png')

# if __name__ == "__main__":
#     test_project_gaussians()
############################################################################################################

import tyro
import os
from core.options import AllConfigs, Options
from core.gs import GaussianRenderer
# from attack_test import AdversarialAttack

def test_extract_and_project_gaussians():
    # 创建随机的3D高斯参数 (N, 14)
    # N = 10000  # 随机点的数量
    # gaussian_params = torch.randn(N, 14, device='cuda:2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = tyro.cli(AllConfigs)
    path = "workspace/0113/"
    gs = GaussianRenderer(opt)
    ply_file = [f for f in os.listdir(path) if f.endswith('.ply')][0]

    gaussian_params = gs.load_ply(os.path.join(path, ply_file)).float().to(device)
    
    # 将位置参数归一化到 [-1, 1] 范围
    gaussian_params[:, 0:3] = torch.tanh(gaussian_params[:, 0:3])
    
    # 实例化 FixedViewTextureProjector 类
    projector = FixedViewTextureProjector()
    
    # 提取五个面的纹理信息
    view_textures = projector.extract_view_textures(gaussian_params)
    
    # 创建十字纹理图像
    cross_texture = projector.create_cross_texture(view_textures)
    
    # # 投影高斯参数到纹理上
    # new_colors = projector.project(gaussian_params, cross_texture)
    
    # # 更新高斯参数中的颜色信息
    # updated_params = projector.apply_to_gaussians(gaussian_params, cross_texture)
    
    # # 打印一些结果以进行验证
    # print("Original Gaussian Params (first 5):", gaussian_params[:5])
    # print("Updated Gaussian Params (first 5):", updated_params[:5])
    
    # 保存生成的十字纹理图像
    cross_texture.save('cross_texture.png')

if __name__ == "__main__":
    test_extract_and_project_gaussians()