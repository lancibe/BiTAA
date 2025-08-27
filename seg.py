import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    deeplabv3_resnet101,  # 新增模型1
    fcn_resnet50,
    fcn_resnet101,        # 新增模型2
    lraspp_mobilenet_v3_large  # 新增模型3
)

# 配置参数
# target_classes = [0, 8, 12]  # COCO类别：0=背景, 8=车, 12=人
# target_classes = [i for i in range (0, 20)]
target_classes = [7]
# target_classes = [3, 6, 8]
image_ids = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
# directory = "./workspace/不同车辆/0304_audi.a2/before_attack"
directory = "./workspace/不同车辆/0304_audi.a2/after_attack"
# image_ids = [0_5_1.5_0, 0_5_1.5_22, 0_5_1.5_20, 0_5_1.5_18, 0_5_1.5_16, 0_5_1.5_14, 
#              0_5_1.5_12, 0_5_1.5_10, 0_5_1.5_8, 0_5_1.5_6, 0_5_1.5_4, 0_5_1.5_2]
# directory = "./core/carla_dataset_full/vehicle.audi.a2/0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化分割模型（增加到四个）
def create_segmentors():
    segmentors = [
        ('deeplabv3_res50', deeplabv3_resnet50(weights='DEFAULT', ).to(device).eval()),
        ('deeplabv3_res101', deeplabv3_resnet101(weights='DEFAULT').to(device).eval()),
        ('fcn_res50', fcn_resnet50(weights='DEFAULT').to(device).eval()),
        ('fcn_res101', fcn_resnet101(weights='DEFAULT').to(device).eval()),
        # ('lraspp_mobilenet', lraspp_mobilenet_v3_large(weights='DEFAULT').to(device).eval())
    ]
    return segmentors

# 统一图像预处理（适配不同模型输入尺寸）
def preprocess_seg(image_path):
    image = Image.open(image_path).convert("RGB")
    
    # 根据不同模型调整尺寸
    transform = T.Compose([
        T.Resize(520 if 'deeplab' in image_path else 256),  # 动态调整
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # print(image.size)
    return transform(image).unsqueeze(0).to(device)

# 处理分割输出（兼容所有模型）
def process_seg_output(output, target_classes):
    if isinstance(output, dict):  # DeepLab/FCN系列
        seg_logits = output['out']
    else:  # LRASPP模型直接输出tensor
        seg_logits = output
    
    # 修正后的置信度计算
    probs = torch.softmax(seg_logits, dim=1)        # 多类别概率
    seg_mask = probs.argmax(dim=1).squeeze().cpu().numpy()  # 预测类别ID

    # 获取目标类别的平均概率
    class_conf = {}
    for cls in target_classes:
        cls_prob_map = probs[0, cls].cpu().numpy()  # 目标类别的概率图
        mask = (seg_mask == cls)                    # 预测为该类别的区域
        if mask.sum() > 0:
            avg_conf = cls_prob_map[mask].mean()    # 区域内平均概率
        else:
            avg_conf = 0.0
        class_conf[cls] = avg_conf
    return class_conf, seg_mask

# 主流程保持不变，兼容所有模型
def main_seg():
    segmentors = create_segmentors()
    stats = {name: {cls: [] for cls in target_classes} for name, _ in segmentors}
    
    for img_id in image_ids:
        path = os.path.join(directory, f"{img_id}.png")
        if not os.path.exists(path):
            continue

        try:
            tensor = preprocess_seg(path)  # 统一预处理
            
            for name, model in segmentors:
                with torch.no_grad():
                    output = model(tensor)
                
                class_conf, _ = process_seg_output(output, target_classes)
                
                for cls in target_classes:
                    stats[name][cls].append(class_conf[cls])
                    
        except Exception as e:
            print(f"处理 {path} 出错: {str(e)}")

    # 打印结果（增加模型对比）
    for name in stats:
        print(f"\n🔍 {name.upper()} 性能")
        for cls in target_classes:
            confs = stats[name][cls]
            print(f"  类别 {cls}:")
            print(f"    ▸ 平均置信: {np.nanmean(confs):.2f}")
            print(f"    ▸ 最大置信: {np.nanmax(confs):.2f}")
            print(f"    ▸ 有效检出: {sum(c > 0.1 for c in confs)}/{len(confs)}张")

if __name__ == "__main__":
    main_seg()