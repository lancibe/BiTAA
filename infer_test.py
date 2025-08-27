# 0, 6, 12, 18``
# CUDA_VISIBLE_DEVICES=2 python infer_test.py big --resume pretrained/model_fp16_fixrot.safetensors --workspace workspace_test_1119 --test_path core/carla_dataset_full/vehicle.audi.a2/0
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
            # # 将 numpy.ndarray 转换为 4D 张量
            # input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
            # input_image = F.interpolate(TF.center_crop(input_image, 540), size=(256, 256), mode='bilinear', align_corners=False)
            # # 将 4D 张量转换回 numpy.ndarray
            # input_image = input_image.squeeze(0).permute(1, 2, 0).byte().numpy()

            # # TODO: resize the images
            # if image_names[i].endswith('_0.png') or image_names[i].endswith('_12.png'):
            #     # Convert the image to torch tensor
            #     input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            #     # Compute the new size (e.g., 50% scale)
            #     scale_factor = 0.5
            #     _, C, H, W = input_tensor.shape
            #     new_size = (int(H * scale_factor), int(W * scale_factor))
            #     # Resize using interpolate
            #     resized_tensor = F.interpolate(input_tensor, size=new_size, mode='bilinear', align_corners=False)
            #     # Create a new canvas with original size and fill with zeros
            #     if C == 4:  # RGBA
            #         padded_tensor = torch.zeros((1, C, H, W), dtype=resized_tensor.dtype)
            #         padded_tensor[:, 3, :, :] = 255  # Set alpha channel to 255 initially
            #     else:  # RGB
            #         padded_tensor = torch.zeros((1, C, H, W), dtype=resized_tensor.dtype)
            #     # Compute padding offsets
            #     y_offset = (H - new_size[0]) // 2
            #     x_offset = (W - new_size[1]) // 2
            #     # Place the resized image in the center of the canvas
            #     padded_tensor[:, :, y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = resized_tensor
                
            #     # If RGBA, set alpha=0 for the blank areas
            #     if C == 4:
            #         padded_tensor[:, 3, :, :] = 0  # Set all alpha to 0
            #         padded_tensor[:, 3, y_offset:y_offset+new_size[0], x_offset:x_offset+new_size[1]] = 255  # Restore resized area to opaque
            #     # Convert back to numpy array and save
            #     input_image = padded_tensor.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

            carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
            mask = carved_image[..., -1] > 0
            image = recenter(carved_image, mask, border_ratio=0.2)
            image = image.astype(np.float32) / 255.0

            # rgba to rgb white bg
            if image.shape[-1] == 4:
                image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])

            input_images.append(image)
    print(f'[INFO] Loaded {len(input_images)} images')
   
    mv_image = np.stack([input_images[2], input_images[3], input_images[0], input_images[1]], axis=0) # [4, 256, 256, 3], float32
    # mv_image = np.stack(input_images, axis=0) # [4, 256, 256, 3], float32

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    # save input_image as a single image
    combined_image = np.concatenate(mv_image, axis=1)  # concatenate along width
    combined_image = (combined_image * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(opt.workspace, f"{name}_combined_image.png"), combined_image)
    # imageio.imwrite(os.path.join(opt.workspace, f"{name}_mv_image.png"), (combined_image * 255).astype(np.uint8))

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 4, dtype=np.int32)
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
