import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import os

plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'axes.titlesize': 15,
    'figure.titlesize': 15
})



# 设置参数
num_angles = 12
num_rings = 3
inner_hole_radius = 0.75  # 圆心留空区域半径
angle_width = 360 / num_angles


# 模拟数据
# np.random.seed(42)
# data = np.random.rand(num_rings, num_angles)
# print("Data shape:", data.shape)
# print("Data values:\n", data)

data = [
    [2.123, 2.262, 3.562, 3.629, 2.839, 1.482, 2.077, 2.451, 3.503, 3.961, 3.189, 1.785],
    [1.829, 2.081, 3.123, 3.405, 2.982, 1.601, 1.908, 2.215, 3.201, 3.789, 2.951, 1.654],
    [1.456, 1.789, 2.901, 3.201, 2.675, 1.234, 1.567, 1.789, 2.901, 3.456, 2.678, 1.345]
]
data = np.array(data)
data = data / 4
print("Data shape:", data.shape)
print("Data values:\n", data)

custom_cmap = LinearSegmentedColormap.from_list("custom_cool", ["#D0CECE","#F8CBAD", "#C55A11" ])


# 创建图形
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(9, 7), facecolor='white')
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4.5, 4.0), facecolor='white')
ax.set_theta_zero_location('S')  # 0° 在南方（下方）
ax.set_theta_direction(1)       # 顺时针方向


# 设置色块偏移角度，使其错开整数角度（±15°）
theta_offset = angle_width / 2
theta_edges = np.linspace(0, 360, num_angles + 1)
r_edges = np.linspace(inner_hole_radius, num_rings + inner_hole_radius, num_rings + 1)

# 绘制每个扇形色块
for i in range(num_rings):
    r0 = r_edges[i]
    r1 = r_edges[i + 1]
    for j in range(num_angles):
        # 色块中心以 j * angle_width 位置为中心（整数刻度），范围 ±15°
        theta_center = j * angle_width
        theta0 = theta_center - theta_offset
        theta1 = theta_center + theta_offset
        val = data[i, j]

        theta_range = np.radians(np.linspace(theta0, theta1, 30))
        r_inner = np.full_like(theta_range, r0)
        r_outer = np.full_like(theta_range, r1)
        # ax.fill_between(theta_range, r_inner, r_outer, color=plt.cm.plasma(val), edgecolor='gray', linewidth=0.3)
        ax.fill_between(theta_range, r_inner, r_outer, color=custom_cmap(val), edgecolor='gray', linewidth=0.3)

# 设置极坐标刻度（固定在整数角度）
angle_ticks = [i * angle_width for i in range(num_angles)]
ax.set_xticks(np.radians(angle_ticks))
ax.set_xticklabels([f'{int(a)}°' for a in angle_ticks])

# 设置距离标签在圆环中心
r_label_pos = [(r_edges[i] + r_edges[i + 1]) / 2 for i in range(num_rings)]
ax.set_yticks(r_label_pos)
ax.set_yticklabels(['Near', 'Mid', 'Far'])

# 去除默认半径虚线
ax.yaxis.grid(False)

# 标题与色条
# ax.set_title("Adversarial Effectiveness by Angle and Distance", va='bottom', fontsize=14, weight='bold')
ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

ax.yaxis.grid(False)  # 去除径向虚线
ax.xaxis.grid(False)  # 去除角度方向虚线

# sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=4))
# sm = plt.cm.ScalarMappable(cmap='cividis', norm=plt.Normalize(vmin=0, vmax=4))
# sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=4))
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=4))

cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label('LCR (Lower → Higher)', fontsize=12)
cbar.ax.tick_params(labelsize=10)


# 放置 logo 图片在图中间（示例使用 matplotlib 自带图像）
logo_path = './车辆-01.png'
if os.path.exists(logo_path):
    image = mpimg.imread(logo_path)
    imagebox = OffsetImage(image, zoom=0.1)
    ab = AnnotationBbox(imagebox, (0, 0), frameon=False, boxcoords="data")
    ax.add_artist(ab)

plt.tight_layout()
plt.savefig('./polar.pdf', format='pdf', bbox_inches='tight')
