import os
import cv2
import random
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui

from scipy.spatial.transform import Rotation
# from kiui.op import safe_normalize
# from kiui.typing import *
from kiui.cam import convert, look_at

from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

# from core.utils import save_rays_as_ply # for test only

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class ObjaverseDataset(Dataset):
    def __init__(self, opt: Options, training=True):
        self.opt = opt
        self.training = training

        self.items = []

        
        # 遍历根目录，收集所有车辆和采样点的信息
        root_dir = opt.data_root  # 数据集的根目录
        vehicles = os.listdir(root_dir)  # 根目录下的所有车辆名称
        for vehicle in vehicles:
            vehicle_path = os.path.join(root_dir, vehicle)
            if os.path.isdir(vehicle_path):
                # 对于每个车辆目录，遍历所有采样点
                sampling_points = os.listdir(vehicle_path)
                for point in sampling_points:
                    point_path = os.path.join(vehicle_path, point)
                    if os.path.isdir(point_path):
                        # 将车辆和采样点的组合加入到 items 列表中
                        self.items.append({
                            'vehicle': vehicle,
                            'point': point,
                            'path': point_path  # 保存采样点的完整路径
                        })

        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:] 


        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        # print(f'default_camera_matrix: {self.proj_matrix}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        point_path = item['path']
        vehicle = item['vehicle']
        point = item['point']
        
        # 读取采样点中的 JSON 文件
        json_path = os.path.join(point_path, 'camera_params.json')
        with open(json_path, 'r') as f:
            camera_info = json.load(f)

        images = []
        masks  = []
        angles = []
        cam_poses = []

        vehicle_type = []
        weather = []
        distance = []
        height = []

        # 遍历 JSON 文件中的 24 张图片信息
        for entry in camera_info:
            image_filename = entry['image_file']
            image_path = os.path.join(point_path, image_filename)

            # 读取 RGBA 图像
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # 读取 RGBA 图像
            image = torch.from_numpy(image.astype(np.float32) / 255)  # 归一化到 [0, 1]
            image = image.permute(2, 0, 1)  # [4, H, W]，将通道放在最前面

            mask = image[3:4]
            image = image[:3] * mask + (1 - mask)  # 去掉 alpha 通道
            # 提取相机位置和旋转信息
            # location = entry['abs location']  # [x, y, z]
            location = entry['location']
            rotation = entry['rotation']  # [pitch, yaw, roll]

            # 将 location 和 rotation 转换为相机矩阵（例如4x4矩阵）
            c2w, radius = self.get_camera_matrix(location = location, rotation = rotation)
            # print(f'image_path: {image_path}, c2w: {c2w}')
            # c2w[:3, 3] *= self.opt.cam_radius / 1.5 # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] /= radius  # normalize the translation to [-1, 1]^3 space

            # 将图像和相机姿态加入到列表中
            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)
            angles.append(entry['angle'])

            vehicle_type.append(vehicle)
            weather.append(entry['weather'])
            distance.append(entry['distance'])
            height.append(entry['height'])

        # 将列表转换为张量
        images = torch.stack(images, dim=0)  # [24, 3, H, W]
        masks = torch.stack(masks, dim=0)  # [24, H, W]
        cam_poses = torch.stack(cam_poses, dim=0)  # [24, 4, 4]

        # 归一化相机姿态
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses # [24, 4, 4] 只包含4个视角


        # 从 24 张图像中选择角度为 0、90、180、270 的图像
        selected_indices = [i for i, angle in enumerate(angles) if angle in [0, 90, 180, 270]]
        images_input = images[selected_indices]
        cam_poses_input = cam_poses[selected_indices]
        masks_input = masks[selected_indices]     

        # 图像大小调整
        images_input = F.interpolate(TF.center_crop(images_input, 540), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)   
        # images_input = F.interpolate(images_input, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)   

        # 归一化相机姿态
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses_input[0])
        # cam_poses_input = transform.unsqueeze(0) @ cam_poses_input  # [4, 4, 4] 只包含4个视角

        # 数据增强（如果需要）
        if self.training:
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        # 图像归一化
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # 光线嵌入计算
        rays_embeddings = []
        for i in range(len(images_input)):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy)  # [H, W, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1)  # [H, W, 6]
            rays_embeddings.append(rays_plucker)
            # for test only
            # save_rays_as_ply(rays_o, rays_d, num_rays=100, save_name=f"{item['path'].split('/')[-1]}+{i}", save_path="./core/output")

        
        # 将光线嵌入堆叠成张量
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous()  # [4, 6, H, W]

        # 构建输入，拼接图像和光线嵌入
        final_input = torch.cat([images_input, rays_embeddings], dim=1)  # [4, 9, H, W]

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1  # invert up & forward direction

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2)  # [4, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix  # [4, 4, 4]
        cam_pos = - cam_poses[:, :3, 3]  # [4, 3]

        results = {
            'input': final_input,
            'images_output': F.interpolate(TF.center_crop(images, 540), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False),  # 原始 24 张图像输出
            'masks_output': F.interpolate(TF.center_crop(masks.unsqueeze(1), 540), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False),  # 原始 24 张掩码输出
            # 'images_output': F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False),
            # 'masks_output': F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False),  # 原始 24 张掩码输出
            # 'cam_poses_input': cam_poses_input,  # 筛选后的相机姿态
            'cam_view': cam_view,  # 视图矩阵
            'cam_view_proj': cam_view_proj,  # 视图-投影矩阵
            'cam_pos': cam_pos,  # 相机位置
            
            'vehicle_model': vehicle_type,  # 从 JSON 读取车辆型号
            'weather': weather,  # 从 JSON 读取天气信息
            'distance': distance,  # 从 JSON 读取距离信息
            'height': height,  # 从 JSON 读取高度信息
            'angle': angles  # 从 JSON 读取角度信息
        }

        return results

    def get_camera_matrix(self, location, rotation, target_system='opengl'):
        p, y, r = np.radians(rotation)
        c = np.cos
        s = np.sin

        elevation = p
        azimuth = y
        rotation_matrix = np.array([
            [c(p)*c(y), c(y)*s(p)*s(r) - c(r)*s(y), -c(y)*s(p)*c(r) - s(y)*s(r)],
            [c(p)*s(y), s(y)*s(p)*s(r) + c(y)*c(r), -s(y)*s(p)*c(r) + c(y)*s(r)],
            [s(p), -c(p)*s(r), c(p)*c(r)]
        ])

        x, y, z = np.array(location, dtype=np.float32)
        radius = np.sqrt(x**2 + y**2 + z**2)

        Rc = rotation_matrix
        To = np.array([radius*c(elevation)*s(azimuth), -radius*s(elevation), radius*c(elevation)*c(azimuth)], dtype=np.float32).T
        M = np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = M @ Rc.T @ M.T 
        c2w[:3, 3] = To

        return torch.tensor(c2w, dtype=torch.float32), radius





if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    opt = Options()

    train_dataset = ObjaverseDataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = ObjaverseDataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # 测试 DataLoader 迭代
    for batch_idx, batch_data in enumerate(train_dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Input shape: {batch_data['input'].shape}")
        print(f"Camera view shape: {batch_data['cam_view'].shape}")
        print(f"Camera view proj shape: {batch_data['cam_view_proj'].shape}")
        print(f"Camera pos shape: {batch_data['cam_pos'].shape}")
        print(f"Output image shape: {batch_data['images_output'].shape}")
        print(f"Output mask shape: {batch_data['masks_output'].shape}")
        
        if batch_idx == 2:  # 测试两个 batch 就可以
            break

    for batch_idx, batch_data in enumerate(test_dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Input shape: {batch_data['input'].shape}")
        print(f"Camera view shape: {batch_data['cam_view'].shape}")
        print(f"Camera view proj shape: {batch_data['cam_view_proj'].shape}")
        print(f"Camera pos shape: {batch_data['cam_pos'].shape}")
        print(f"Output image shape: {batch_data['images_output'].shape}")
        print(f"Output mask shape: {batch_data['masks_output'].shape}")
        
        if batch_idx == 2:  # 测试两个 batch 就可以
            break