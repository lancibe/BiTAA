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

from physical import PhysicalAugmentation

# ==== NEW: extra imports ====
from typing import Dict, Tuple
import torch.fft

# image helpers
def chw01(img):  # (1,1,3,H,W) or (3,H,W) -> (3,H,W) in [0,1]
    if img.dim()==5:
        img = img.squeeze(0).squeeze(0)   # (3,H,W)
    return img.clamp(0,1)

def tv_loss_2d(x):
    # x: (C,H,W) in [0,1]
    dh = (x[...,1:,:] - x[...,:-1,:]).abs().mean()
    dw = (x[...,:,1:] - x[...,:,:-1]).abs().mean()
    return dh + dw



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

# ==== NEW: Monocular depth wrapper ====
class DepthEstimator:
    """
    统一接口：.predict(image_chw_in_0_1) -> depth_chw(H,W) in meters or pseudo-meters
    你可以替换内部实现为 ZoeDepth / DPT-Large。
    """
    def __init__(self, device):
        self.device = device
        self.model, self.transform = None, None
        try:
            # 方案A：MiDaS DPT-Large（如果环境可用）
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.dpt_transform
            print("[INFO] Depth model: MiDaS DPT-Large.")
        except Exception as e:
            print("[WARN] MiDaS not available:", e)
            # 方案B：占位符（请用你的模型替换下面两行）
            self.model = None
            self.transform = None
            print("[WARN] DepthEstimator is a stub. Please plug in your ZoeDepth/DPT!")

    @torch.no_grad()
    def predict_nograd(self, img_chw01: torch.Tensor) -> torch.Tensor:
        """仅用于生成 d0(干净参考)，不反传。"""
        if self.model is None:
            # 占位：返回亮度的假深度，便于打通管线（请尽快替换）
            fake = img_chw01.mean(dim=0, keepdim=True)  # (1,H,W)
            return (fake + 1e-6)
        img = img_chw01
        if self.transform is not None:
            # 变换期望 HWC&RGB，故转下
            img_hwc = img.permute(1,2,0).cpu().numpy()
            net_in = self.transform(img_hwc).to(self.device)
            pred = self.model(net_in)  # (1,H,W)
            return pred.clamp_min(1e-6)
        else:
            return (img.mean(dim=0, keepdim=True) + 1e-6)

    def predict(self, img_chw01: torch.Tensor) -> torch.Tensor:
        """可反传（不对模型反传，只对输入和上游参数反传）。"""
        # 不对 self.model 反传：用 no_grad 得到网络输出，再当作常数参与损失会断梯度；
        # 为了对上游可微（渲染→像素→损失），这里采用“停止对网络求导”的做法：
        with torch.no_grad():
            d = self.predict_nograd(img_chw01)  # (1,H,W), detached
        return d  # 上游的梯度仍可通过损失 w.r.t img_chw01 回传到3DGS


class AdversarialAttack:
    def __init__(self, device, opt):
        self.device = device
        self.opt = opt

        self.detector = fasterrcnn_resnet50_fpn(pretrained=True, threshold=1e-5).to(device)
        print("[INFO] Detector fasterrcnn_resnet50_fpn loaded successfully.")

        # self.detector = maskrcnn_resnet50_fpn(pretrained=True).to(device)
        # print("[INFO] Detector maskrcnn_resnet50_fpn loaded successfully.")

        # self.detector = ssd300_vgg16(pretrained=True, threshold=1e-5).to(device)
        # print("[INFO] Detector ssd300_vgg16 loaded successfully.")
        
        # self.detector = YOLOv3Detector(device)
        # print("[INFO] Detector YOLOv3 loaded successfully.")
        self.detector.eval()  # 设置为评估模式, YOLOv3不需要设置为评估模式
        self.gs = GaussianRenderer(opt)

        # ==== NEW: depth model & caches ====
        self.depth_model = DepthEstimator(device)

        # hyper-params (可从 opt 里覆写)
        self.bias_dir = getattr(opt, "bias_dir", +1)         # s in {+1,-1}, +1推远, -1拉近
        self.beta     = getattr(opt, "beta", 0.10)           # log域偏置步长
        self.lambda_det   = getattr(opt, "lambda_det", 1.0)
        self.lambda_dep   = getattr(opt, "lambda_dep", 1.0)
        self.lambda_shape = getattr(opt, "lambda_shape", 0.2)
        self.lambda_print = getattr(opt, "lambda_print", 0.1)

        # 可打印性阈值（RGB参数变化限制，仅示意，可按需调）
        self.rgb_eps_inf = getattr(opt, "rgb_eps_inf", 0.20)

        # 缓存干净参考（每个视角一份）
        self.clean_cache: Dict[int, Dict[str, torch.Tensor]] = {}  # {view_id: {"img":(3,H,W), "depth":(1,H,W)}}


    def load_gaussian(self, path, compatible=True):
        return self.gs.load_ply(path, compatible)

    # def compute_shape_loss(self, generated_gaussians, target_gaussians):
    #     # 计算形状损失
    #     shape_diff = torch.cat((generated_gaussians[:, :3] - target_gaussians[:, :3], generated_gaussians[:, 4:11] - target_gaussians[:, 4:11]), dim=1)
    #     return torch.norm(shape_diff)
    def compute_shape_loss(self, generated_gaussians, target_gaussians, knn_idx=None):
        """
        在你原有 L2 基础上加一小项“局部一致性”，不改变整体风格。
        knn_idx: 可选 (N,k) 的近邻索引；没有就只用原有项。
        """
        base = torch.cat((
            generated_gaussians[:, :3] - target_gaussians[:, :3],      # pos
            generated_gaussians[:, 4:11] - target_gaussians[:, 4:11]   # scale (3) + quat (4)
        ), dim=1)
        l_base = torch.norm(base)

        if knn_idx is None:
            return l_base

        # 局部一致性（对 Δpos 轻微平滑）
        delta_p = generated_gaussians[:, :3] - target_gaussians[:, :3]   # (N,3)
        i = torch.arange(delta_p.shape[0], device=delta_p.device).unsqueeze(-1)  # (N,1)
        nbr = delta_p[knn_idx]  # (N,k,3)
        center = delta_p[i]     # (N,1,3)
        l_smooth = ((center - nbr)**2).mean()
        return l_base + 0.05 * l_smooth  # 系数很小，尽量不“动大”


    def compute_adversarial_loss(self, image):
        # print("Adversarial image requires_grad:", image.requires_grad)
        outputs = self.detector(image)
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

    # ==== NEW: build ROI mask from detector outputs ====
    def build_roi_from_outputs(self, outputs, H, W, score_th=0.2, shrink=0.15):
        mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        target_classes = [3, 6, 8]  # 你的目标类（car, bus, truck等）
        for out in outputs:
            boxes = out['boxes']
            labels = out['labels']
            scores = out['scores']
            for box, lab, sc in zip(boxes, labels, scores):
                if (lab.item() in target_classes) and (sc.item() >= score_th):
                    x1, y1, x2, y2 = box.round().long()
                    x1 = x1.clamp(0, W-1); x2 = x2.clamp(0, W-1)
                    y1 = y1.clamp(0, H-1); y2 = y2.clamp(0, H-1)
                    # 内缩，避免边界噪声
                    bw = (x2 - x1).item(); bh = (y2 - y1).item()
                    x1 = int(x1 + shrink * bw / 2); x2 = int(x2 - shrink * bw / 2)
                    y1 = int(y1 + shrink * bh / 2); y2 = int(y2 - shrink * bh / 2)
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2+1, x1:x2+1] = True
        return mask  # (H,W) bool

    # ==== NEW: log-domain bias loss & Δsig ====
    def depth_bias_loss(self, img_adv_chw01: torch.Tensor, img_clean_chw01: torch.Tensor, roi_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        img_*: (3,H,W) in [0,1], requires_grad for adv image
        roi_mask: (H,W) bool
        return: (L_bias, delta_sig_mean)
        """
        # 干净参考深度 d0：缓存 or 现算（不反传）
        H, W = img_adv_chw01.shape[-2], img_adv_chw01.shape[-1]

        with torch.no_grad():
            d0 = self.depth_model.predict_nograd(img_clean_chw01)  # (1,H,W)

        d_hat = self.depth_model.predict(img_adv_chw01)            # (1,H,W), detached from model but keeps upstream grads
        eps = 1e-6
        log_bias = (d_hat.clamp_min(eps).log() - d0.clamp_min(eps).log() - self.bias_dir * self.beta)  # (1,H,W)

        if roi_mask is None or roi_mask.sum()==0:
            # 无ROI时可退化为整图，或返回0
            roi_mask = torch.ones((H,W), dtype=torch.bool, device=self.device)

        L_bias = (log_bias[:, roi_mask] ** 2).mean()

        delta_sig = (d_hat.clamp_min(eps).log() - d0.clamp_min(eps).log())  # (1,H,W)
        delta_sig_mean = delta_sig[:, roi_mask].mean().detach()

        return L_bias, delta_sig_mean

    # ==== NEW: printability loss ====
    def printability_loss(self, img_adv_chw01: torch.Tensor, img_clean_chw01: torch.Tensor,
                          gauss_adv: torch.Tensor, gauss_orig: torch.Tensor,
                          use_spectral=False) -> torch.Tensor:
        """
        - RGB param budget (on gaussians): L_inf 超额惩罚
        - Image residual smoothness: TV (可选 Spectral高频惩罚)
        """
        # 1) 高斯 RGB 预算（参数域）：(N,14) -> (N,3)
        rgb_adv = gauss_adv[..., 11:14]
        rgb_org = gauss_orig[..., 11:14]
        delta_rgb = (rgb_adv - rgb_org)  # [-?,?]
        l_inf_excess = torch.relu(delta_rgb.abs().amax() - self.rgb_eps_inf)  # 标量

        # 2) 图像残差（像素域）：TV + 可选频域
        resid = (img_adv_chw01 - img_clean_chw01).detach() + (img_adv_chw01 - img_clean_chw01) * 0  # 保持梯度 w.r.t img_adv
        l_tv = tv_loss_2d(resid)

        if not use_spectral:
            return l_inf_excess + 0.1 * l_tv  # 系数可调
        else:
            # 高频能量惩罚：简易圆环mask
            B = resid
            F = torch.fft.fftshift(torch.fft.fft2(B, dim=(-2,-1)))
            mag = (F.real**2 + F.imag**2).sqrt()  # (C,H,W)
            H, W = mag.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(H, device=mag.device), torch.arange(W, device=mag.device), indexing='ij')
            cy, cx = H//2, W//2
            r = torch.sqrt((yy-cy)**2 + (xx-cx)**2).float()
            r_norm = r / r.max()
            hi_mask = (r_norm > 0.5).float()  # 0.5之外为高频
            l_spec = (mag * hi_mask).mean()
            return l_inf_excess + 0.1 * l_tv + 0.01 * l_spec  # 系数可调


    def perform_attack(self, gaussians, generated_gaussians, step):
        azimuth = np.arange(0, 360, 30, dtype=np.int32)
        input_elevation = -10

        total_det_loss = 0.0
        total_bias_loss = 0.0
        total_print_loss = 0.0
        total_conf = 0.0
        num_views = 0
        delta_sig_list = []

        for azi_idx, azi in enumerate(azimuth):
            # --- 视角准备 ---
            cam_poses = torch.from_numpy(orbit_camera(input_elevation, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0).to(self.device)
            cam_poses[:, :3, 1:3] *= -1
            cam_view = torch.inverse(cam_poses).transpose(1, 2)
            cam_view_proj = cam_view @ proj_matrix
            cam_pos = - cam_poses[:, :3, 3]

            # --- 渲染（adv & clean）---
            if generated_gaussians.dim() == 2:
                generated_gaussians = generated_gaussians.unsqueeze(0)
            generated_gaussians.retain_grad()

            raw_adv = self.gs.render(generated_gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)
            img_adv = chw01(raw_adv['image'])
            alpha = chw01(raw_adv['alpha'])

            # clean 只渲染一次并缓存
            view_id = int(azi)  # key
            if view_id not in self.clean_cache:
                raw_clean = self.gs.render(gaussians.unsqueeze(0), cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)
                img_clean = chw01(raw_clean['image']).detach()
                self.clean_cache[view_id] = {"img": img_clean}
            else:
                img_clean = self.clean_cache[view_id]["img"]

            # --- 可选：保存渲染 ---
            if (step + 1) % 10 == 0:
                save_path = f"./{self.opt.workspace}/render_outputs/step_{step}_azi_{azi_idx}.png"
                self.save_image(raw_adv['image'], alpha, save_path)

            # --- 物理增强（如需对齐同一变换，可扩展 augmentor 支持“返回参数并复用”）---
            augmentor = PhysicalAugmentation()
            # 检测：在增强后图像上
            img_det_in = augmentor.augment(img_adv.permute(1,2,0)).permute(2,0,1).unsqueeze(0)  # (1,3,H,W)

            # 深度偏置：为了稳定，这里使用“未增强”的两张对照（也可以增强后两张都同参）
            img_bias_adv = img_adv
            img_bias_clean = img_clean

            # --- 检测前向 & ROI ---
            outputs = self.detector(img_det_in)[0:1]  # list, len=1
            # 你的原始目标置信度下压
            target_classes = [3, 6, 8]
            is_target = torch.isin(outputs[0]['labels'], torch.tensor(target_classes, device=self.device))
            if torch.any(is_target):
                conf = outputs[0]['scores'][is_target].max()
                det_loss = -torch.log(torch.clamp(conf, min=1e-6))
                total_conf += conf.item()
            else:
                det_loss = torch.tensor(10.0, device=self.device, requires_grad=True)

            # ROI mask
            H, W = img_adv.shape[-2], img_adv.shape[-1]
            roi_mask = self.build_roi_from_outputs(outputs, H, W, score_th=0.2, shrink=0.15)

            # --- 深度偏置损失（log域A2）---
            bias_loss, delta_sig = self.depth_bias_loss(img_bias_adv, img_bias_clean, roi_mask)
            delta_sig_list.append(delta_sig.item())

            # --- 可打印性损失 ---
            print_loss = self.printability_loss(img_adv, img_clean, generated_gaussians[0], gaussians)

            # --- 汇总 ---
            total_det_loss = total_det_loss + det_loss
            total_bias_loss = total_bias_loss + bias_loss
            total_print_loss = total_print_loss + print_loss
            num_views += 1

        # 平均
        if num_views == 0:
            avg_det_loss = torch.tensor(10.0, device=self.device, requires_grad=True)
            avg_bias_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_print_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            avg_conf = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            avg_det_loss = total_det_loss / num_views
            avg_bias_loss = total_bias_loss / num_views
            avg_print_loss = total_print_loss / num_views
            avg_conf = torch.tensor(total_conf / num_views, device=self.device)

        # 形状稳定
        shape_loss = self.compute_shape_loss(generated_gaussians[0], gaussians)

        # 总损失
        total_loss = ( self.lambda_det   * avg_det_loss
                     + self.lambda_dep   * avg_bias_loss
                     + self.lambda_shape * shape_loss
                     + self.lambda_print * avg_print_loss )

        # 日志
        mean_delta_sig = (sum(delta_sig_list)/len(delta_sig_list)) if len(delta_sig_list)>0 else 0.0
        print(f"Step: {step} | Det: {avg_det_loss.item():.4f} | Bias: {avg_bias_loss.item():.4f} | Print: {avg_print_loss.item():.4f} | Shape: {shape_loss.item():.4f} | Conf: {avg_conf.item():.4f} | Δsig: {mean_delta_sig:+.4f}")

        with open(f"./{self.opt.workspace}/attack_log.txt", "a") as f:
            f.write(f"{step}, {shape_loss.item():.6f}, {avg_det_loss.item():.6f}, {avg_conf.item():.6f}, {avg_bias_loss.item():.6f}, {avg_print_loss.item():.6f}, {mean_delta_sig:.6f}\n")

        return total_loss


    # def perform_attack(self, gaussians, generated_gaussians, step):
    #     # 2. 生成12个视角
    #     azimuth = np.arange(0, 360, 30, dtype=np.int32)
    #     input_elevation = -10

    #     total_adv_loss = 0
    #     total_conf = 0
    #     num_views = len(azimuth)

    #     for azi_idx, azi in enumerate(azimuth):
    #         # 生成相机位置信息
    #         cam_poses = torch.from_numpy(orbit_camera(input_elevation, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0).to(self.device)
    #         cam_poses[:, :3, 1:3] *= -1

    #         cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
    #         cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
    #         cam_pos = - cam_poses[:, :3, 3] # [V, 3]

    #         if generated_gaussians.dim() == 2:
    #             generated_gaussians = generated_gaussians.unsqueeze(0)
    #         generated_gaussians.retain_grad()

    #         image_raw = self.gs.render(generated_gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)

    #         image = image_raw['image']
    #         alpha = image_raw['alpha']

    #         # 保存当前渲染的图像
    #         if step+1 % 10 == 0:
    #             save_path = f"./{self.opt.workspace}/render_outputs/step_{step}_azi_{azi_idx}.png"
    #             self.save_image(image, alpha, save_path)

    #         # 3. 物理增强
    #         # if step % 2 == 0:
    #         image = image.squeeze(0).squeeze(0).permute(1, 2, 0)
    #         augmentor = PhysicalAugmentation()
    #         image = augmentor.augment(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

    #         # 4. 计算损失
    #         adv_loss, conf = self.compute_adversarial_loss(image[0])
    #         conf = conf.item() if isinstance(conf, torch.Tensor) else conf

    #         if adv_loss.item() != 10.0:
    #             total_adv_loss += adv_loss
    #             total_conf += conf
    #         else:
    #             num_views -= 1

    #     # print(num_views)
    #     avg_adv_loss = (total_adv_loss / num_views) if num_views > 0 else torch.tensor(10.0, device=self.device, requires_grad=True)
    #     avg_conf = (total_conf / num_views) if num_views > 0 else torch.tensor(0.0, device=self.device, requires_grad=True)

    #     shape_loss = self.compute_shape_loss(generated_gaussians[0], gaussians)

    #     # shape_loss = shape_loss * 0.0

    #     adv_scale_factor = 10.0  
    #     min_adv_weight = 0.2
    #     min_shape_weight = 0.4
    #     scaled_adv_loss = avg_adv_loss / adv_scale_factor 
    #     raw_adv_weight = 1.0 / (scaled_adv_loss.detach() + 1e-6) # 计算动态权重（直接使用损失值）
         
    #     raw_shape_weight = shape_loss.detach()
    #     total_raw_weight = raw_adv_weight + raw_shape_weight + 1e-6
    #     adv_weight = raw_adv_weight / total_raw_weight
    #     shape_weight = raw_shape_weight / total_raw_weight

    #     # shape_weight = 0.0
    #     # adv_weight = torch.clamp(adv_weight, min=min_adv_weight)
    #     # shape_weight = 1.0 - adv_weight
    #     shape_weight = torch.clamp(shape_weight, min=min_shape_weight)
    #     adv_weight = 1.0 - shape_weight
    #     total_loss = -adv_weight * avg_adv_loss + shape_weight * shape_loss

    #     print(f"Step: {step}, Shape Loss: {shape_loss.item()}, Adversarial Loss: {avg_adv_loss.item()}, Confidence: {avg_conf}")
        
    #     with open(f"./{self.opt.workspace}/attack_log.txt", "a") as f:
    #         f.write(f"{step}, {shape_loss.item()}, {avg_adv_loss.item()}, {avg_conf}\n")

    #     return total_loss


def process(opt: Options, path):
    attack = AdversarialAttack(device, opt)
    ply_file = [f for f in os.listdir(path) if f.endswith('.ply')][0]

    gaussians = attack.load_gaussian(os.path.join(path, ply_file)).float().to(device)
    generated_gaussians = gaussians.clone().detach().requires_grad_(True).to(device)


    optimizer = torch.optim.Adam([generated_gaussians], lr=0.0001)

    # 冻结的维度
    mask = torch.zeros_like(generated_gaussians).to(device)
    # mask[:, 0:3] = 1 # 保留 xyz 维度
    # mask[:, 3] = 1 # 保留 a 维度
    # mask[:, 4:7] = 1 # 保留 s 维度
    # mask[:, 7:11] = 1 # 保留 q 维度
    # mask[:, 11:14] = 1  # 保留 RGB 维度
    mask[:, :] = 1
    # mask = torch.ones_like(generated_gaussians).to(device)

    # 训练循环
    for i in range(50):
        optimizer.zero_grad()  # 每步迭代清除梯度

        # 执行攻击并返回损失
        loss = attack.perform_attack(gaussians, generated_gaussians, i)
        
        # 计算梯度
        loss.backward()

        # 应用掩码屏蔽冻结的维度
        if generated_gaussians.grad is not None:
            generated_gaussians.grad *= mask

        # 执行优化步骤
        optimizer.step()

    # 保存攻击后的高斯模型
    output_ply_file = os.path.splitext(ply_file)[0]
    attack.gs.save_ply(generated_gaussians.unsqueeze(0), os.path.join(path, output_ply_file + '_attack.ply'), compatible=True)

    # 计算3D高斯的变化
    pos_orig, pos_adv = gaussians[..., :3], generated_gaussians[..., :3]
    opacity_orig, opacity_adv = gaussians[..., 3], generated_gaussians[..., 3]
    scale_orig, scale_adv = gaussians[..., 4:7], generated_gaussians[..., 4:7]
    rotation_orig, rotation_adv = gaussians[..., 7:11], generated_gaussians[..., 7:11]
    rgbs_orig, rgbs_adv = gaussians[..., 11:14], generated_gaussians[..., 11:14]

    # 计算欧式距离（位置和旋转）
    pos_change = torch.norm(pos_adv - pos_orig, dim=-1).mean()  # 位置变化
    rotation_change = torch.norm(rotation_adv - rotation_orig, dim=-1).mean()  # 旋转变化

    # 计算均方误差（MSE）
    opacity_change = torch.mean((opacity_adv - opacity_orig) ** 2)  # 透明度变化
    scale_change = torch.mean((scale_adv - scale_orig) ** 2)  # 尺度变化
    color_change = torch.mean((rgbs_adv - rgbs_orig) ** 2)  # 颜色变化

    print(f'Position Change: {pos_change.item():.6f}')
    print(f'Opacity Change: {opacity_change.item():.6f}')
    print(f'Scale Change: {scale_change.item():.6f}')
    print(f'Rotation Change: {rotation_change.item():.6f}')
    print(f'Color Change: {color_change.item():.6f}') 

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


    # 在指定位置单独保存每一帧
    for i, img in enumerate(images):
        # if i % 15 != 0:
        #     continue
        os.makedirs(os.path.join(opt.workspace, "after_attack"), exist_ok=True)
        imageio.imwrite(os.path.join(os.path.join(opt.workspace, "after_attack"), f"{i:03d}.png"), img[0])

    images = np.concatenate(images, axis=0)
    imageio.mimwrite(os.path.join(path, output_ply_file + '_attack.mp4'), images, fps=30)
    print("[INFO] Done Attack!")


assert opt.workspace is not None
print("[INFO] Start attack processing...")
process(opt, opt.workspace)
