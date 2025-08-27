import numpy as np
import os
import tyro
import imageio
import warnings
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # 作为示例目标检测模型
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import ssd300_vgg16
from yolo import YOLOv3Detector

from PIL import Image

import kiui
from kiui.cam import orbit_camera

from core.unet import UNet
from core.gs import GaussianRenderer
from core.options import AllConfigs, Options

opt = tyro.cli(AllConfigs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

class AdversarialAttack:
    def __init__(self, device, opt):
        self.device = device
        self.opt = opt

        # self.detector = fasterrcnn_resnet50_fpn(pretrained=True, threshold=1e-5).to(device)
        # print("[INFO] Detector fasterrcnn_resnet50_fpn loaded successfully.")

        # self.detector = maskrcnn_resnet50_fpn(pretrained=True).to(device)
        # print("[INFO] Detector maskrcnn_resnet50_fpn loaded successfully.")

        self.detector = ssd300_vgg16(pretrained=True, threshold=1e-5).to(device)
        print("[INFO] Detector ssd300_vgg16 loaded successfully.")
        
        # self.detector = YOLOv3Detector(device)
        # print("[INFO] Detector YOLOv3 loaded successfully.")
        self.detector.eval()  # 设置为评估模式, YOLOv3不需要设置为评估模式
        self.gs = GaussianRenderer(opt)

    def load_gaussian(self, path, compatible=True):
        # 使用load_ply加载3D高斯模型
        return self.gs.load_ply(path, compatible)

    def compute_shape_loss(self, generated_gaussians, target_gaussians):
        # 计算形状损失
        # 假设只比较x, s, q的维度
        shape_diff = torch.cat((generated_gaussians[:, :3] - target_gaussians[:, :3], generated_gaussians[:, 4:11] - target_gaussians[:, 4:11]), dim=1)
        return torch.norm(shape_diff)
        # # 暂时返回一个零值
        # return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_adversarial_loss(self, image):
        # print("Adversarial image requires_grad:", image.requires_grad)
        outputs = self.detector(image)
        # outputs = self.detector.forward(image) # for YOLOv3
        # target_classes = [2, 3, 5, 7] 
        target_classes = [3, 6, 8]# [2, 7]  
        conf_scores = []
        for output in outputs:
            # 检查是否有目标类别存在
            # import pdb; pdb.set_trace()
            is_target = torch.isin(output['labels'], torch.tensor(target_classes, device=output['labels'].device))
            if torch.any(is_target):
                # 提取目标类别的置信度
                target_scores = output['scores'][is_target]
                conf_scores.append(torch.max(target_scores))  # 取当前目标中的最大置信度

        if conf_scores:
            conf_scores = torch.stack(conf_scores) 
            return -torch.log(torch.clamp(torch.mean(conf_scores), min=1e-6)), conf_scores
        else:
            return torch.tensor(10.0, device=self.device, requires_grad=True), "None"
        # return torch.tensor(0.0, device=self.device, requires_grad=True)

    def save_image(self, image, alpha, output_path):
        # 去掉 Batch 和 View 维度
        image_tensor = image.squeeze(0).squeeze(0)  # 从 (1, 1, 3, 512, 512) 到 (3, 512, 512)
        alpha_tensor = alpha.squeeze(0).squeeze(0)  # 从 (1, 1, 1, 512, 512) 到 (1, 512, 512)

        # 转换为 (H, W, C) 格式
        image_array = image_tensor.permute(1, 2, 0).detach().cpu().numpy()  # (3, 512, 512) -> (512, 512, 3)
        alpha_array = alpha_tensor.permute(1, 2, 0).detach().cpu().numpy()  # (1, 512, 512) -> (512, 512, 1)

        # 将 [0, 1] 的浮点数值转换为 [0, 255] 的 uint8 类型
        image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
        alpha_array = (alpha_array * 255).clip(0, 255).astype(np.uint8)

        # 拼接 RGB 和 Alpha 通道
        rgba_array = np.concatenate((image_array, alpha_array), axis=-1)

        # 使用 PIL 保存为 RGBA 格式图像
        rgba_image = Image.fromarray(rgba_array, 'RGBA')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 创建文件夹
        rgba_image.save(output_path)
        # print(f"RGBA image saved to {output_path}")


    def perform_attack(self, gaussians, optimizer, step):
        # 2. 生成12个视角
        azimuth = np.arange(0, 360, 30, dtype=np.int32)
        input_elevation = -10

        total_adv_loss = 0
        total_conf = 0
        num_views = len(azimuth)

        for azi_idx, azi in enumerate(azimuth):
            # 生成相机位置信息
            cam_poses = torch.from_numpy(orbit_camera(input_elevation, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0).to(self.device)
            cam_poses[:, :3, 1:3] *= -1

            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            # 确保 generated_gaussians 的形状符合 render 方法的预期
            generated_gaussians = optimizer.param_groups[0]['params'][0].float()  # 从优化器中获取叶张量
            if generated_gaussians.dim() == 2:
                generated_gaussians = generated_gaussians.unsqueeze(0)
            generated_gaussians.retain_grad()

            # mask = torch.zeros_like(generated_gaussians, requires_grad=False).to(self.device)
            # mask[:, :, 11:14] = 1  # 保留 rgb 维度
            # mask[:, :, 0:3] = 1

            # # 冻结除了rgb以外的所有维度
            # generated_gaussians[:, :, 0:11].requires_grad = False

            image_raw = self.gs.render(generated_gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)

            image = image_raw['image']
            alpha = image_raw['alpha']

            # 保存当前渲染的图像
            if step % 10 == 0:
                save_path = f"./{self.opt.workspace}/render_outputs/step_{step}_azi_{azi_idx}.png"
                self.save_image(image, alpha, save_path)

            # 3. 物理增强
            image = self.apply_physical_augmentation(image)
            # print("Image shape:", image.shape) # torch.Size([1, 1, 3, 512, 512])

            # 4. 计算损失
            adv_loss, conf = self.compute_adversarial_loss(image[0])
            conf = conf.item() if isinstance(conf, torch.Tensor) else conf

            if adv_loss.item() != 10.0:
                total_adv_loss += adv_loss
                total_conf += conf
            else:
                num_views -= 1

        # print(num_views)
        avg_adv_loss = (total_adv_loss / num_views) if num_views > 0 else torch.tensor(10.0, device=self.device, requires_grad=True)
        avg_conf = (total_conf / num_views) if num_views > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)

        shape_loss = self.compute_shape_loss(generated_gaussians[0], gaussians)

        print(f"Shape Loss: {shape_loss.item()}, Adversarial Loss: {avg_adv_loss.item()}, Confidence: {avg_conf}")

        # 5. 更新高斯参数
        adv_weight = 1.0 / (avg_adv_loss.detach().item() + 1e-6)
        shape_weight = 1.0 / (shape_loss.detach().item() + 1e-6)
        total_weight = adv_weight + shape_weight
        adv_weight /= total_weight
        shape_weight /= total_weight
        total_loss = -adv_weight * avg_adv_loss + shape_weight * shape_loss

        optimizer.zero_grad()
        total_loss.backward()
        # print("Gradients before masking:", generated_gaussians.grad[:, :11]) 
        # generated_gaussians.grad *= mask # 通过这一行代码控制某些维度的梯度不更新
        # print("Gradients after masking:", generated_gaussians.grad[:, :11]) 
        optimizer.step()

    def apply_physical_augmentation(self, image):
        # 物理增强处理逻辑
        # image.size() = torch.Size([1, 1, 3, 512, 512])
        # 模仿高曝光和高阴影，随机二选一进行增强
        if np.random.rand() < 0.5:
            image = image + 0.2
        else:
            image = image - 0.2

        # 模仿相机畸变和部分遮挡
        # 生成随机遮挡区域
        # mask = torch.zeros_like(image)
        


        return image  


def process(opt: Options, path):
    attack = AdversarialAttack(device, opt)
    ply_file = [f for f in os.listdir(path) if f.endswith('.ply')][0]
    gaussians = attack.load_gaussian(os.path.join(path, ply_file)).float().to(device)
    generated_gaussians = gaussians.clone().to(device).requires_grad_(True)

    # print("Gradients before masking:", generated_gaussians.grad)
    # mask = torch.zeros_like(generated_gaussians).to(device)
    # mask[:, 11:14] = 1  # 保留 rgb 维度
    # generated_gaussians.grad *= mask  # 冻结除了rgb以外的所有维度
    # print("Gradients after masking:", generated_gaussians.grad) 

    optimizer = torch.optim.Adam([generated_gaussians], lr=0.0001)
    # print("Requires grad:", generated_gaussians.requires_grad)

    # target_gaussians = torch.randn(1, 14)  # 假设的目标高斯数据
    # 进行迭代攻击
    for i in range(100):
        # print("Optimizer params:", optimizer.param_groups[0]['params'][0])
        print(f"Step {i + 1}: ", end='\t')
        attack.perform_attack(gaussians, optimizer, i)
        # print("Gradient is:", generated_gaussians.grad)  

    # 保存攻击后的高斯模型
    output_ply_file = os.path.splitext(ply_file)[0]
    attack.gs.save_ply(generated_gaussians.unsqueeze(0), os.path.join(path, output_ply_file + '_attack.ply'), compatible=True)
    # for i in range(10):
    #     adv_loss = attack.compute_shape_loss(generated_gaussians, gaussians)  # 使用最简单的损失函数
    #     optimizer.zero_grad()
    #     adv_loss.backward()
    #     print("Gradient after backward:", generated_gaussians.grad)  # 检查梯度
    #     optimizer.step()

    # render 360 video 
    images = []
    elevation = 0

    azimuth = np.arange(0, 720, 2, dtype=np.int32)
    for azi in tqdm.tqdm(azimuth):
        
        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

        scale = min(azi / 360, 1)

        image = attack.gs.render(generated_gaussians.unsqueeze(0), cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
        images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().detach().numpy() * 255).astype(np.uint8))

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(os.path.join(path, output_ply_file + '_attack.mp4'), images, fps=30)
    print("[INFO] Done Attack!")


assert opt.workspace is not None
print("[INFO] Start attack processing...")
process(opt, opt.workspace)
