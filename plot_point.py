# 重新导入所需库
import matplotlib.pyplot as plt
import pandas as pd

# 示例输入数据
data = [
    [75.42,  0.1749, "UPC", "2D Attack"],
    [60.12,  0.4590, "DAS", "2D Attack"],
    [70.21,  0.1829, "CAMOU", "3D Attack"],
    [35.89,  0.2987, "FCA", "3D Attack"],
    [47.44,  0.1804, "DTA", "3D Attack"],
    [33.56,  0.1814, "ACTIVE", "3D Attack"],
    [40.12,  0.3876, "TT3D", "3D Attack"],
    [7.38,   0.4951, "3DGAA", "Ours"]
]

# 转为 DataFrame
df = pd.DataFrame(data, columns=["x", "y", "name", "category"])

# 自定义颜色
color_dict = {
    "2D Attack": "#4472c4",
    "3D Attack": "#f8cbad",
    "Ours": "#ff0000"
}

# 分类对应的 marker 形状
marker_dict = {
    "2D Attack": "o",  # 圆圈
    "3D Attack": "s",  # 方块
    "Ours": "*"   # 五角星
}

# 创建图像
fig, ax = plt.subplots(figsize=(6, 4))

# 遍历类别并绘图
for cat in df["category"].unique():
    sub_df = df[df["category"] == cat]
    marker_size = 750 if cat == "Ours" else 100
    if cat == "Ours":
        ax.scatter(
            sub_df["x"], sub_df["y"],
            color=color_dict[cat],
            marker=marker_dict[cat],
            label='_nolegend_',
            s=marker_size, edgecolors='black'
        )
    else:
        ax.scatter(
            sub_df["x"], sub_df["y"],
            color=color_dict[cat],
            marker=marker_dict[cat],
            label=cat,
            s=marker_size, edgecolors='black'
        )
    for _, row in sub_df.iterrows():
        if cat == "Ours":
            # 对于 "Ours" 类别，标签位置稍微上移
            ax.text(row["x"], row["y"] + 0.025, row["name"], ha='center', fontsize=16)  
        else:
            ax.text(row["x"], row["y"] + 0.015, row["name"], ha='center', fontsize=16)

    # # 添加标签
    # for _, row in sub_df.iterrows():
    #     ax.text(row["x"], row["y"] + 0.015, row["name"], ha='center', fontsize=16)

ax.scatter([], [], color=color_dict["Ours"], marker="*", label="Ours", s=150, edgecolors='black')

# 坐标轴设置
ax.set_xlim(100, 0)  # 横坐标反向
ax.set_ylim(0.15, 0.55)
ax.set_xlabel("Adversarial Robustness (mAP)", fontsize=20)
ax.set_ylabel("Physical Realism (PSNR)", fontsize=20)
ax.legend(title="Category", fontsize=12,  title_fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# # 设置x/y刻度，不包含最大值
# x_ticks = list(range(0, 101, 20))
# x_ticks.remove(0)
# ax.set_xticks(x_ticks)
# y_ticks = [0.15, 0.25, 0.35, 0.45]
# ax.set_yticks(y_ticks)

# 隐藏默认坐标轴
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# 用annotate画箭头坐标轴
arrowprops = dict(arrowstyle="->", linewidth=1.5, color='black')
# x轴箭头
ax.annotate('', xy=(0, 0.15), xytext=(100, 0.15), arrowprops=arrowprops, annotation_clip=False)
# y轴箭头
ax.annotate('', xy=(100, 0.55), xytext=(100, 0.15), arrowprops=arrowprops, annotation_clip=False)


# 保存为PDF（无损）
output_path = "./plot_point.pdf"
plt.tight_layout()
plt.savefig(output_path, format='pdf')
# plt.show()

# output_path
