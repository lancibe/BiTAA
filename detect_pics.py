import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.ops import nms
from torchvision.io import read_image
from torchvision.models.detection import *
from torchvision.transforms import functional as F
from ultralytics import YOLO

import matplotlib.pyplot as plt  

# 配置参数
target_classes = [3, 6, 8]  # COCO类别ID：3=摩托车，6=公共汽车，8=卡车
# image_ids = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
# directory = "./workspace/不同车辆/0304_toyota.prius/after_attack"
# directory = "./data_test/对抗/"
# directory = "./data_test/无强化对抗/"
# directory = "./data_test/原图/"
# directory = './exps/pics/temp/'
directory = './exps/0306/'
# directory = './workspace/0307/before_attack/'
# directory = './workspace/0307/after_attack/'
# output_dir = "./detection_results/真实对抗_0702"
# output_dir = "./detection_results/真实对抗_0719"
# output_dir = "./detection_results/无强对抗_0719"
# output_dir = "./detection_results/原始_0719"
output_dir = "./detection_results/visualization"
# confidence_threshold = 0.70
confidence_threshold = 0.3
nms_threshold = 0.3
car_class_id = 3  # COCO中的car类别ID
# car_class_id = [3, 8]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# COCO类别名称映射
coco_names = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle',
    5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
    9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella',
    31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard',
    42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife',
    50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot',
    58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake',
    62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    66: 'dining table', 67: 'toilet', 68: 'tv', 69: 'laptop',
    70: 'mouse', 71: 'remote', 72: 'keyboard', 73: 'cell phone',
    74: 'microwave', 75: 'oven', 76: 'toaster', 77: 'sink',
    78: 'refrigerator', 79: 'book', 80: 'clock', 81: 'vase',
}

def create_detectors():
    detectors = [
        # ('fasterrcnn', fasterrcnn_resnet50_fpn(pretrained=True).to(device).eval()),
        ('maskrcnn', maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()),
        # ('ssd300', ssd300_vgg16(pretrained=True).to(device).eval()),
        # ('yolov3', YOLO('yolov3.pt').to(device)),
        # ('yolov5', YOLO('yolov5s.pt').to(device)),
        # ('yolov8', YOLO('yolov8n.pt').to(device))
    ]
    return detectors

def read_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    # image = image.crop((int(original_size[0]/3), int(original_size[1]/3),
    #                 int(original_size[0]*2/3), int(original_size[1]*2/3)))
    new_size = image.size
    return image, new_size

def preprocess(image_path, model_type):
    # image = Image.open(image_path).convert("RGB")
    # original_size = image.size
    image, original_size = read_image(image_path)
    
    if not model_type.startswith('yolo'):
        if model_type == 'ssd300':
            image = F.resize(image, (300, 300))
            tensor = F.to_tensor(image).unsqueeze(0).to(device)
        else:
            # image = F.resize(image, (512, 512))
            tensor = F.to_tensor(image).unsqueeze(0).to(device)
        return tensor, original_size
    else:
        return np.array(image), original_size

def process_output(output, model_name, original_size):
    detections = []
    if 'rcnn' in model_name or 'ssd' in model_name:
        boxes = output[0]['boxes'].cpu().detach().numpy()
        labels = output[0]['labels'].cpu().detach().numpy()
        scores = output[0]['scores'].cpu().detach().numpy()
        
        if model_name == 'ssd300':
            orig_w, orig_h = original_size
            scale_x = orig_w / 300
            scale_y = orig_h / 300
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
        
        for i in range(len(scores)):
            detections.append({
                'box': boxes[i],
                'label': labels[i],
                'score': scores[i]
            })
    elif model_name.startswith('yolo'):
        boxes = output.boxes.xyxy.cpu().numpy()
        labels = output.boxes.cls.cpu().numpy().astype(int)
        scores = output.boxes.conf.cpu().numpy()
        for i in range(len(scores)):
            detections.append({
                'box': boxes[i],
                'label': labels[i],
                'score': scores[i]
            })
    return detections

def apply_nms(detections, iou_threshold):
    if not detections:
        return []
    
    # 转换为tensor
    boxes = torch.tensor([d['box'] for d in detections])
    scores = torch.tensor([d['score'] for d in detections])
    
    # 应用NMS
    keep_idx = nms(boxes, scores, iou_threshold)
    
    return [detections[i] for i in keep_idx]

def draw_boxes(image, detections):
    img = image.copy()
    has_car = False
    has_valid = False

    confidence_text = ""  # 新增：用于存储置信度文本

    # 只保留置信度最高的检测结果
    if detections:
        # 按置信度排序并应用NMS
        sorted_detections = sorted(
            [d for d in detections if d['score'] >= confidence_threshold],
            key=lambda x: x['score'], 
            reverse=True
        )
        filtered_detections = apply_nms(sorted_detections, nms_threshold)
        
        # 如果有多个检测结果，只保留置信度最高的
        if filtered_detections:
            best_det = max(filtered_detections, key=lambda x: x['score'])
            box = best_det['box'].astype(int)
            label = best_det['label']
            score = best_det['score']

            # 确定颜色
            color = (0, 0, 255)  # 默认红色
            # if label in car_class_id:
            if label == car_class_id:
                color = (0, 255, 0)  # 绿色
                has_car = True
            has_valid = True

            # 绘制检测框
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # 添加标签
            label_text = f"{coco_names.get(label, str(label))} {score:.2f}"
            # font_scale = 1.8  # 字体大小
            # thickness = 2     # 线宽
            font_scale = 1.5  # 字体大小
            thickness = 2     # 线宽
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(img, (box[0], box[1]-th-2), (box[0]+tw, box[1]), color, -1)
            cv2.putText(img, label_text, (box[0], box[1]-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
            
            # 新增：记录置信度文本
            confidence_text = f"Confidence: {score:.2f}"

    # 新增：将置信度写到图片左上角
    # if confidence_text:
    #     cv2.putText(img, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


    # 确定边框颜色
    border_color = (0, 0, 0)
    if has_car:
        border_color = (0, 255, 0)
    elif has_valid:
        border_color = (0, 0, 255)
        
    # 绘制图像边框
    border_size = 5
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, border_size), border_color, -1)
    cv2.rectangle(img, (0, h-border_size), (w, h), border_color, -1)
    cv2.rectangle(img, (0, 0), (border_size, h), border_color, -1)
    cv2.rectangle(img, (w-border_size, 0), (w, h), border_color, -1)

    return img

def main():
    detectors = create_detectors()
    stats = {name: [] for name, _ in detectors}
    
    os.makedirs(output_dir, exist_ok=True)
    # 修改为遍历所有图片
    # for img_id in os.listdir(directory):
    #     if img_id.endswith(".png") or img_id.endswith(".jpg"):
    #         path = os.path.join(directory, img_id)
    # for img_id in image_ids:
    for root, dirs, files in os.walk(directory):
        for img_id in files:
            if not img_id.endswith(".png") and not img_id.endswith(".jpg"):
                continue
            # print(dirs)
            path = os.path.join(root, f"{img_id}")
            if not os.path.exists(path):
                print(f"Image path does not exist: {path}")
                continue
                
            relative_path = os.path.relpath(root, directory)
            output_subdir = os.path.join(output_dir, relative_path)
            os.makedirs(output_subdir, exist_ok=True)  # 自动创建目录结构
            
            original_image = cv2.imread(path)
            if original_image is None:
                continue

            for name, detector in detectors:
                try:
                    # 预处理
                    processed_input, orig_size = preprocess(path, name)
                    
                    # 推理
                    if name.startswith('yolo'):
                        results = detector(processed_input)
                        output = results[0]
                    else:
                        with torch.no_grad():
                            output = detector(processed_input)
                    
                    # 处理输出
                    detections = process_output(output, name, orig_size)
                    
                    # 绘制并保存
                    result_img = draw_boxes(original_image, detections)
                    # save_dir = os.path.join(output_dir, name)
                    # os.makedirs(save_dir, exist_ok=True)
                    # cv2.imwrite(os.path.join(save_dir, f"{img_id}.png"), result_img)
                    # 修改保存路径：使用新目录结构
                    save_path = os.path.join(output_subdir, img_id)
                    # cv2.imwrite(save_path, result_img)
                    if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
                        cv2.imwrite(save_path, result_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    else:
                        plt.imsave(save_path, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), dpi=800)
                    # save_path_pdf = os.path.splitext(save_path)[0] + '.pdf'
                    # plt.imsave(save_path_pdf, cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), dpi=600)

                    # 统计
                    valid_scores = [d['score'] for d in detections 
                                if d['score'] >= confidence_threshold 
                                and d['label'] in target_classes]                
                    stats[name].append(max(valid_scores) if valid_scores else 0)
                    
                except Exception as e:
                    print(f"Error in {name} for {img_id}: {e}")
                    stats[name].append(0)

    # 输出统计结果
    for name, values in stats.items():
        if values:
            print(f"\n{name.upper()}:")
            print(f"Max: {max(values):.4f}  Min: {min(values):.4f}  Avg: {np.mean(values):.4f}")

if __name__ == "__main__":
    main()