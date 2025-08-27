import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np
import os
from scipy.signal import savgol_filter


plt.rcParams.update({
    'font.size': 12,
    # 'font.family': 'Arial',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11
})



def plot_lcr_curves(file_paths, output_path="lcr_plot.png", labels = None):
    """
    绘制带断点的LCR曲线图
    
    :param file_paths: 日志文件路径列表
    :param output_path: 输出图片保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 修正坐标范围格式为列表嵌套元组
    bax = brokenaxes(
        xlims=[(0, 60), (380, 400)],  # 修正为列表格式
        ylims=[(0, 4)],               # 增加列表包裹
        hspace=0.05,
        despine=False
    )

    colors = plt.cm.tab10.colors
    linestyles = ['-', '-']
    markers = ['o', 'o']
    
    for file_idx, file_path in enumerate(file_paths):
        # 读取数据
        epochs, confidences = [], []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                epochs.append(int(parts[0]))
                confidences.append(float(parts[-1]))
        
        # 计算LCR
        conf_array = np.array(confidences)
        conf_array[conf_array < 0.01] = 0.015
        max_conf = np.max(conf_array)
        # max_conf = conf_array[0]
        lcr = np.log(max_conf / conf_array)
        # print(lcr)
        
        lcr_smooth = savgol_filter(lcr, window_length=15, polyorder=2)

        x = np.array(epochs)
        if len(epochs) > 50:  # 长数据映射
            mask = x > 60
            x[mask] = 380 + (x[mask] - 60) * (20/340) # 线性映射
        
        # 绘制主曲线
        line = bax.plot(
            x, lcr_smooth,
            color=colors[file_idx % len(colors)],
            linestyle=linestyles[file_idx % len(linestyles)],
            label=labels[file_idx] if labels else os.path.basename(file_path),
            linewidth=2,
            zorder=3
        )
        # print(line[0][0])
        line_color = line[0][0].get_color()
        # 查找最大值点
        max_lcr = np.max(lcr_smooth)
        max_idx = np.argmax(lcr_smooth)
        raw_epoch = epochs[max_idx]
        mapped_x = x[max_idx]

        # 确定绘制坐标轴
        target_ax = bax.axs[0] if raw_epoch <= 60 else bax.axs[1]
        
        # 绘制最大值标记
        target_ax.scatter(
            mapped_x, max_lcr,
            color=line_color,
            marker=markers[file_idx % len(markers)],
            s=80,
            edgecolor='white',
            zorder=5
        )

        # # 绘制水平引导线
        # target_ax.hlines(
        #     y=max_lcr,
        #     # xmin=0 if raw_epoch <=60 else 380,
        #     xmin=0,
        #     xmax=mapped_x,
        #     colors=line_color,
        #     linestyles='dashed',
        #     linewidth=1,
        #     alpha=0.7,
        #     zorder=4
        # )

        # 添加数值标注
        target_ax.text(
            mapped_x + (2 if raw_epoch <=60 else -2),  # 偏移量调整
            max_lcr,
            f'{max_lcr:.2f}',
            color=line_color,
            # fontsize=10,
            fontsize=16,
            va='center',
            ha='left' if raw_epoch <=60 else 'right',
            # ha='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )


    # 坐标轴美化
    left_ax, right_ax = bax.axs
    for ax in [left_ax, right_ax]:
        ax.grid(True, alpha=0.3)
    
    # 设置右边坐标轴刻度
    right_ax.set_xticks([380, 390, 400])
    right_ax.set_xticklabels(['60', '200', '400'])  # 显示映射后的实际值
    
    # 全局标签
    bax.set_xlabel("Adversarial Optimization Epoch", fontsize=20, labelpad=20) # fontsize=12
    bax.set_ylabel("Log Confidence Reduction (LCR)", fontsize=20, labelpad=25)
    bax.legend(loc="lower right", fontsize=12, framealpha=0.6) # fontsize=8
    
    # 保存图像
    # plt.legend()
    plt.savefig(output_path, dpi=800, bbox_inches='tight')
    plt.close()

# 使用示例
if __name__ == "__main__":
    files = [
        # "path/to/short_data.log",  # 50个epoch
        # "path/to/long_data.log"    # 400个epoch
        './workspace/维度选择/0303_all/attack_log.txt',
        # './workspace/维度选择/0303_tex/attack_log.txt',
        # './workspace/维度选择/0303_geo/attack_log.txt'
        './workspace/维度选择/0303_x/attack_log.txt',
        './workspace/维度选择/0303_s/attack_log.txt',
        './workspace/维度选择/0303_q/attack_log.txt',
        './workspace/维度选择/0303_c/attack_log.txt',
        './workspace/维度选择/0303_a/attack_log.txt',
    ]

    labels = [
        'All',
        'x',
        's',
        'q',
        'c',
        'α'
    ]
    # labels = [
    #     'All',
    #     'Appearance',
    #     'Geometry',
    #     # 'q',
    #     # 'c',
    #     # 'α'
    # ]
    # plot_lcr_curves(files, "lcr_plot.png", labels)
    plot_lcr_curves(files, "lcr_plot.pdf", labels)

    # def create_test_data():
    #     # 短数据 (50 epochs)
    #     with open("short.log", "w") as f:
    #         for i in range(50):
    #             f.write(f"{i},0.1,0.2,{0.9 - i*0.015}\n")
        
    #     # 长数据 (400 epochs)
    #     with open("long.log", "w") as f:
    #         for i in range(400):
    #             f.write(f"{i},0.1,0.2,{0.9 - i*0.002}\n")

    # create_test_data()
    # plot_lcr_curves(["short.log", "long.log"], "test.png")