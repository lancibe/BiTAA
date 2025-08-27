import torch
from lpips import LPIPS
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 LPIPS（感知距离）模型
lpips_loss = LPIPS(net='vgg').to(device)

# 图像预处理
def preprocess_image(img):
    # 如果是RGBA格式，转换为RGB
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    # 把图像转换为512x512大小
    
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0).to(device)

# 计算 LPIPS（感知损失）
def compute_lpips(img1, img2):
    img1_tensor = preprocess_image(img1)
    img2_tensor = preprocess_image(img2)
    return lpips_loss(img1_tensor, img2_tensor).item()

# 计算 SSIM（结构相似度）
def compute_ssim(img1, img2):
    img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
    return ssim(img1_gray, img2_gray, data_range=255)

# 计算 PSNR（峰值信噪比）
def compute_psnr(img1, img2):
    # 确保图像模式一致
    if img1.mode != img2.mode:
        img2 = img2.convert(img1.mode)
    return psnr(np.array(img1), np.array(img2), data_range=255)

# 计算所有对比项
def evaluate_images(ply_image, before_image, after_image):
    metrics = {
        "LPIPS_image1_vs_ply": compute_lpips(before_image, ply_image),
        "LPIPS_image2_vs_ply": compute_lpips(after_image, ply_image),
        # "LPIPS_before_vs_after": compute_lpips(before_image, after_image),
        
        "SSIM_image1_vs_ply": compute_ssim(before_image, ply_image),
        "SSIM_image2_vs_ply": compute_ssim(after_image, ply_image),
        # "SSIM_before_vs_after": compute_ssim(before_image, after_image),

        "PSNR_image1_vs_ply": compute_psnr(before_image, ply_image),
        "PSNR_image2_vs_ply": compute_psnr(after_image, ply_image),
        # "PSNR_before_vs_after": compute_psnr(before_image, after_image),
    }
    
    return metrics


# 新增统计函数
def calculate_statistics(metrics_list):
    statistics = defaultdict(list)
    
    # 收集所有数据
    for metrics in metrics_list:
        for key, value in metrics.items():
            statistics[key].append(value)
    
    # 计算统计量
    results = {}
    for key, values in statistics.items():
        results[key] = {
            "mean": np.mean(values),
            "max": np.max(values),
            "min": np.min(values),
            "std": np.std(values)
        }
    return results

def exp1():
    # 示例：传入三张渲染图像
    # ply_image = Image.open("./core/carla_dataset_full/vehicle.audi.a2/0/0_5_1.5_0.png")  # PLY 渲染
    # image1 = Image.open("./workspace/0302_none/before_attack/180.png")  
    # # image2 = Image.open("./workspace/0302_3D/before_attack/180.png")  
    # image2 = Image.open("./workspace/0302_full/before_attack/180.png")
    # ply_image = Image.open("./core/carla_dataset_full/vehicle.audi.a2/0/0_5_1.5_2.png")  # PLY 渲染
    # image1 = Image.open("./exps/pics/t1.png")  
    # image2 = Image.open("./exps/pics/t2.png")
    # ply_image = Image.open("./core/carla_dataset_full/vehicle.audi.a2/0/0_5_1.5_23.png")  # PLY 渲染
    # image1 = Image.open("./exps/pics/t3.png")  
    # image2 = Image.open("./exps/pics/t4.png")
    ply_image = Image.open("./core/carla_dataset_full/vehicle.audi.a2/0/0_5_1.5_23.png")  # PLY 渲染
    # image1 = Image.open("./exps/pics/t5.png")  
    image1 = Image.open("./core/carla_dataset_full/vehicle.audi.a2/0/0_5_1.5_23.png")
    image2 = Image.open("./exps/pics/t6.png")

    ply_image = ply_image.resize((512, 512))
    image1 = image1.resize((512, 512))
    image2 = image2.resize((512, 512))
    print(ply_image.size, image1.size, image2.size)
    # 计算所有指标
    metrics = evaluate_images(ply_image, image1, image2)
    print(metrics)

def exp2():
    import os
    # 示例2：传入三个目录，每个目录中有一系列图片，按照定好的列表顺序进行对比
    ply_list = [0, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
    before_list = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    after_list = [180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

    ply_dir = "./core/carla_dataset_full/vehicle.audi.a2/0"
    before_dir = "./workspace/0303_1/before_attack"
    after_dir = "./workspace/0303_1/after_attack"

    metrics_list = []
    for ply_idx, before_idx, after_idx in zip(ply_list, before_list, after_list):
        ply_image = Image.open(os.path.join(ply_dir, f"0_5_1.5_{ply_idx}.png"))
        before_image = Image.open(os.path.join(before_dir, f"{before_idx}.png"))
        after_image = Image.open(os.path.join(after_dir, f"{after_idx}.png"))
        ply_image = ply_image.resize((512, 512))
        before_image = before_image.resize((512, 512))
        after_image = after_image.resize((512, 512))
        metrics = evaluate_images(ply_image, before_image, after_image)
        metrics_list.append(metrics)

    print(metrics_list)

    # 计算并打印统计结果
    stats = calculate_statistics(metrics_list)
    print("\n========== 统计结果 ==========")
    for metric_name, values in stats.items():
        print(f"{metric_name}:")
        print(f"  均值: {values['mean']:.6f}")
        print(f"  最大值: {values['max']:.6f}")
        print(f"  最小值: {values['min']:.6f}")
        print(f"  标准差: {values['std']:.6f}\n")


if __name__ == "__main__":
    exp1()
    # exp2()
