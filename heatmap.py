import cv2
import numpy as np
import matplotlib.pyplot as plt

def heatmap_spatial(img1, img2):
    # 计算绝对差值
    diff = cv2.absdiff(img1, img2)

    # 归一化到 [0,255]
    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 生成热力图
    heatmap = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET)

    return heatmap

def heatmap_frequency(img1, img2):
    # 计算傅里叶变换
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    
    # 取对数幅度谱
    magnitude1 = np.log(np.abs(np.fft.fftshift(f1)) + 1)
    magnitude2 = np.log(np.abs(np.fft.fftshift(f2)) + 1)
    
    # 计算差异
    freq_diff = np.abs(magnitude1 - magnitude2)
    
    # 归一化并生成热力图
    norm_diff = cv2.normalize(freq_diff, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap

# 读取图像
img1 = cv2.imread('./workspace/0302_none/before_attack/180.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('./workspace/0302_full/before_attack/180.png', cv2.IMREAD_GRAYSCALE)
# img1 = cv2.imread('./exps/pics/with_filter.png', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('./exps/pics/without_filter.png', cv2.IMREAD_GRAYSCALE)

# 四通道转三通道
# if img1.shape[2] == 4:
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
# if img2.shape[2] == 4:
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))

frequency = heatmap_frequency(img1, img2)
spatial = heatmap_spatial(img1, img2)

alpha = 1  # 权重调整
final_heatmap = cv2.addWeighted(spatial, alpha, frequency, 1 - alpha, 0)

# 保存结果
cv2.imwrite('heatmap.jpg', final_heatmap)
