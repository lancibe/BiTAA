import os
# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.transforms.functional import to_tensor

# 设置虚拟显示（在无图形界面环境下）
os.environ["DISPLAY"] = ":0"

# 加载预训练的SSD模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = SSD300_VGG16_Weights.COCO_V1  # 或 SSD300_VGG16_Weights.DEFAULT
detector = ssd300_vgg16(weights=weights, score_thresh=1e-5).to(device)
detector.eval()

# 主流程
today_dir = datetime.now().strftime("%m%d")
image_dir = os.path.join(os.path.join("./workspace", today_dir), "render_outputs")

# 检查目录是否存在
if not os.path.exists(image_dir):
    print(f"[ERROR] Directory {image_dir} does not exist.")
    exit(1)

# 列出目录中的所有图像文件
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
if not image_files:
    print(f"[ERROR] No images found in {image_dir}.")
    exit(1)

print(f"[INFO] Found {len(image_files)} images for detection.")

# 定义目标类别的名称（COCO 数据集）
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# 创建保存检测结果的文件夹
output_dir = os.path.join(image_dir, "detections")
os.makedirs(output_dir, exist_ok=True)

# 开始检测
results = []
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"[INFO] Processing {image_file}...")

    # 加载图像
    image = plt.imread(image_path)
    if image.shape[-1] == 4:  # RGBA -> RGB
        image = image[..., :3]

    # 转为张量并移动到设备
    image_tensor = to_tensor(image).unsqueeze(0).to(device)

    # 运行目标检测
    with torch.no_grad():
        detections = detector(image_tensor)

    # 解析检测结果
    if detections:
        detection = detections[0]
        boxes = detection['boxes'].cpu().numpy()
        labels = detection['labels'].cpu().numpy()
        scores = detection['scores'].cpu().numpy()

        # 保存结果
        detected_objects = []
        for box, label, score in zip(boxes, labels, scores):
            if score >= 0.5:  # 设置置信度阈值
                class_name = class_names[label - 1]  # COCO类别从1开始
                detected_objects.append((class_name, score))
                # print(f"Detected {class_name} with confidence {score:.4f}")

        # 保存到结果列表
        results.append({
            "image": image_file,
            "detections": detected_objects
        })

        # 绘制检测结果并保存
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        for box, label, score in zip(boxes, labels, scores):
            if score >= 0.5:
                class_name = class_names[label - 1]
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f"{class_name}: {score:.2f}", color='white', fontsize=10, backgroundcolor='red')
        plt.axis('off')
        output_path = os.path.join(output_dir, f"detection_{image_file}")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"[INFO] Detection result saved to {output_path}")

# 打印总结
print(f"[INFO] Detection completed. Results saved to {output_dir}")