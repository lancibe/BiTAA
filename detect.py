import os
import torch
import cv2
import numpy as np
from torchvision.io import read_image
from torchvision.models.detection import *
from torchvision.transforms import functional as F
from PIL import Image
from ultralytics import YOLO

# 配置参数
target_classes = [3, 6, 8]  # COCO类别ID：3=摩托车，6=公共汽车，8=卡车
image_ids = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]      # 要处理的图片ID
# directory = "./workspace/不同检测器/0304_ssd/after_attack"      # 图片目录路径
directory = "./workspace/0304/before_attack"      # 图片目录路径
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化所有检测器
def create_detectors():
    detectors = [
        ('fasterrcnn', fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()),
        ('maskrcnn', maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()),
        ('ssd300', ssd300_vgg16(pretrained=True).to(device).eval()),
        ('yolov3', YOLO('yolov3.pt').to(device)),
        ('yolov5', YOLO('yolov5s.pt').to(device)),
        ('yolov8', YOLO('yolov8n.pt').to(device))
    ]
    return detectors

# 图像预处理
def preprocess(image_path, model_type):
    image = Image.open(image_path).convert("RGB")

    # TorchVision模型预处理
    if not model_type.startswith('yolo'):
        return F.to_tensor(image).unsqueeze(0).to(device)
    
    return np.array(image)

# 处理检测结果
def process_output(output, model_name):
    if 'rcnn' in model_name or 'ssd' in model_name:
        scores = output[0]['scores'].cpu().detach().numpy()
        labels = output[0]['labels'].cpu().detach().numpy()
        return labels, scores
    # 处理YOLO输出
    if isinstance(output, list):  # YOLOv5/v8格式
        results = output[0].boxes
        return results.cls.cpu().numpy(), results.conf.cpu().numpy()
    elif hasattr(output, 'xyxyn'):  # YOLOv3格式
        return output.xyxyn[0][:, -1].cpu().numpy(), output.xyxyn[0][:, 4].cpu().numpy()

# 主处理流程
def main():
    detectors = create_detectors()
    stats = {name: [] for name, _ in detectors}  # 存储各模型置信度
    
    for img_id in image_ids:
        path = os.path.join(directory, f"{img_id}.png")
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue

        for name, detector in detectors:
            try:
                # 前向传播
                tensor = preprocess(path, name)
                with torch.no_grad():
                    output = detector(tensor)
                
                # 处理输出
                labels, scores = process_output(output, name)
                
                # 筛选目标类别
                valid_scores = [s for l, s in zip(labels, scores) if l in target_classes]
                max_score = max(valid_scores) if valid_scores else 0
                stats[name].append(max_score)
                
            except Exception as e:
                print(f"Error processing {path} with {name}: {str(e)}")
                stats[name].append(0)

    # 统计结果
    for name, values in stats.items():
        if values:
            print(f"\n==== {name.upper()} ====")
            print(f"Max Confidence: {np.max(values):.4f}")
            print(f"Min Confidence: {np.min(values):.4f}")
            print(f"Avg Confidence: {np.mean(values):.4f}")

if __name__ == "__main__":
    main()