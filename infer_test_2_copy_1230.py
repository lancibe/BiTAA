# 0, 6, 12, 18``
# CUDA_VISIBLE_DEVICES=2 python infer_test_2.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace workspace_test_1119 --test_path core/carla_dataset_full/vehicle.audi.a2/0
# vehicle.toyota.prius, vehicle.mini.cooper_s, vehicle.tesla.cybertruck, vehicle.seat.leon
# vehicle.nissan.micra, vehicle.micro.microlino, vehicle.citroen.c3, vehicle.audi.a3
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg
from PIL import Image
from scipy.spatial import KDTree


import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

rays_embeddings = model.prepare_default_rays(device, elevation=-10)
# rays_embeddings = model.prepare_default_rays_24(device, elevation=-10)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# # load image dream
# pipe = MVDreamPipeline.from_pretrained(
#     "ashawkey/imagedream-ipmv-diffusers", # remote weights
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     # local_files_only=True,
# )
# pipe = pipe.to(device)

import cv2

def uniform_resize(image, target_size=(256, 256), border_ratio=0.2):
    """
    Ensure consistent resizing of the object within the image.
    :param image: Input image with alpha channel [H, W, 4].
    :param target_size: Target size (height, width).
    :param border_ratio: Ratio of border padding around the object.
    :return: Resized image.
    """
    # Calculate the object boundary using alpha channel
    mask = image[..., -1] > 0  # Alpha channel as the mask
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return cv2.resize(image, target_size)  # No object, just resize

    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()

    # Add border padding
    height, width = image.shape[:2]
    y_pad = int((y_max - y_min) * border_ratio)
    x_pad = int((x_max - x_min) * border_ratio)

    y_min = max(0, y_min - y_pad)
    y_max = min(height, y_max + y_pad)
    x_min = max(0, x_min - x_pad)
    x_max = min(width, x_max + x_pad)

    # Crop and resize the image
    cropped_image = image[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LINEAR)

    # Handle RGBA to RGB conversion with a white background
    if resized_image.shape[-1] == 4:
        alpha = resized_image[..., 3:] / 255.0  # Normalize alpha
        resized_image = resized_image[..., :3] * alpha + (1 - alpha) * 255.0  # Blend with white background

    return resized_image.astype(np.uint8)

def post_process_gaussians_auto_threshold(gaussians, percentile=80, scale_factor=16/9):
    device = gaussians.device
    gaussians_cpu = gaussians.to('cpu').numpy()
    # 提取各部分数据
    means3D = gaussians_cpu[0, :, 0:3]
    opacity = gaussians_cpu[0, :, 3:4]
    scales = gaussians_cpu[0, :, 4:7]
    rotations = gaussians_cpu[0, :, 7:11]
    shs = gaussians_cpu[0, :, 11:]

    # 1. 自动计算去噪阈值
    def compute_dynamic_threshold(points, percentile):
        """
        根据点的最近邻距离分布，动态计算去噪阈值。
        """
        tree = KDTree(points)
        distances, _ = tree.query(points, k=2)  # 查找每个点的最近邻距离
        dynamic_threshold = np.percentile(distances[:, 1], percentile)  # 取最近邻距离的 p 百分位
        return dynamic_threshold, distances[:, 1]  # 返回阈值和所有距离

    # 计算动态阈值
    dynamic_threshold, all_distances = compute_dynamic_threshold(means3D, percentile)
    # 过滤噪声点
    noise_mask = all_distances < dynamic_threshold  # 保留最近邻距离小于阈值的点

    # 应用 mask 移除噪声点
    means3D = means3D[noise_mask]
    opacity = opacity[noise_mask]
    scales = scales[noise_mask]
    rotations = rotations[noise_mask]
    shs = shs[noise_mask]

    # 2. y 轴放大处理
    means3D[:, 2] *= scale_factor  
    scales[:, 2] *= scale_factor  

    # 合并处理后的数据
    processed_gaussians = np.concatenate([
        means3D,
        opacity,
        scales,
        rotations,
        shs
    ], axis=1)[np.newaxis, :, :]  # 重新扩展为 [1, N, 13] 的格式

    # 转回 PyTorch 张量并恢复到 CUDA
    processed_gaussians_tensor = torch.tensor(processed_gaussians, dtype=torch.float32, device=device)

    return processed_gaussians_tensor, dynamic_threshold

# load rembg
bg_remover = rembg.new_session()

# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    # 循环读取四张图片
    input_images = []
    image_names = os.listdir(path)
    for i in range(len(image_names)):
        # if image_names[i].endswith('.png'):
        if image_names[i].endswith('_0.png') or image_names[i].endswith('_6.png') or image_names[i].endswith('_12.png') or image_names[i].endswith('_18.png'):
        # if image_names[i].endswith('_3.png') or image_names[i].endswith('_9.png') or image_names[i].endswith('_15.png') or image_names[i].endswith('_21.png'):
            input_image = kiui.read_image(os.path.join(path, image_names[i]), mode='uint8')

            carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
            processed_image = uniform_resize(carved_image, target_size=(256, 256), border_ratio=0.2)
            # Normalize the image
            normalized_image = processed_image.astype(np.float32) / 255.0
            input_images.append(normalized_image)

    print(f'[INFO] Loaded {len(input_images)} images')
   
    mv_image = np.stack([input_images[2], input_images[3], input_images[0], input_images[1]], axis=0) # [4, 256, 256, 3], float32
    # mv_image = np.stack(input_images, axis=0) # [4, 256, 256, 3], float32

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # save input_image as a single image
    combined_image = np.concatenate(mv_image, axis=1)  # concatenate along width
    imageio.imwrite(os.path.join(opt.workspace, f"{name}_combined_image.png"), (combined_image * 255).astype(np.uint8))
    # imageio.imwrite(os.path.join(opt.workspace, f"{name}_mv_image.png"), (combined_image * 255).astype(np.uint8))

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # post process gaussians
        gaussians, dynamic_threshold = post_process_gaussians_auto_threshold(gaussians)
        print(f'[INFO] Dynamic threshold: {dynamic_threshold:.4f}')

        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, name + '.mp4'), images, fps=30)


assert opt.test_path is not None
# if os.path.isdir(opt.test_path):
#     file_paths = glob.glob(os.path.join(opt.test_path, "*"))
# else:
#     file_paths = [opt.test_path]
# for path in file_paths:
#     process(opt, path)

process(opt, opt.test_path)

# print(f'[INFO] Done!')