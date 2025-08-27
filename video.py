import cv2
import os
import numpy as np

# 配置参数（关键修正）
CORNER_SIZE = 2
GREEN_THRESHOLD = 1
PURE_GREEN = np.array([0, 255, 0], dtype=np.uint8)  # 必须使用numpy数组

class GreenCounter:
    def __init__(self, title: str, img_w: int, img_h: int):
        self.count = 0
        self.title = title
        # 移除尺寸验证（允许小尺寸）
        
        # 定义检测区域（使用绝对坐标）
        self.regions = [
            (0, 0, CORNER_SIZE, CORNER_SIZE),
            (img_w-CORNER_SIZE, 0, CORNER_SIZE, CORNER_SIZE),
            (0, img_h-CORNER_SIZE, CORNER_SIZE, CORNER_SIZE),
            (img_w-CORNER_SIZE, img_h-CORNER_SIZE, CORNER_SIZE, CORNER_SIZE)
        ]
        # 确保坐标不越界
        self.regions = [
            (max(0,x), max(0,y), min(w, img_w-x), min(h, img_h-y)) 
            for (x,y,w,h) in self.regions
        ]

    def check_green(self, img: np.ndarray) -> bool:
        for (x, y, w, h) in self.regions:
            if w <=0 or h <=0:  # 跳过无效区域
                continue
                
            roi = img[y:y+h, x:x+w]
            mask = cv2.inRange(roi, PURE_GREEN, PURE_GREEN)  # 使用numpy数组
            if cv2.countNonZero(mask) >= GREEN_THRESHOLD:
                self.count += 1
                return True
        return False

    def draw_info(self, img):
        """在图片上绘制标题和统计信息"""
        # 绘制标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 1.0
        title_thickness = 2
        (title_w, title_h), _ = cv2.getTextSize(self.title, font, title_scale, title_thickness)
        title_x = int((img.shape[1] - title_w) / 2)
        title_y = 60  # 标题垂直位置
        
        # 标题背景
        cv2.rectangle(img, 
                     (title_x-10, title_y-title_h-10),
                     (title_x + title_w + 10, title_y+10),
                     (0, 0, 0), -1)
        
        # 标题文字
        cv2.putText(img, self.title,
                   (title_x, title_y),
                   font, title_scale, (255, 255, 255),
                   title_thickness, cv2.LINE_AA)
        
        # 统计信息
        count_text = f"Detected: {self.count}"
        (text_w, text_h), _ = cv2.getTextSize(count_text, font, 0.8, 2)
        cv2.putText(img, count_text,
                   (20, img.shape[0]-30),  # 左下角显示
                   font, 0.8, (0, 255, 0),
                   2, cv2.LINE_AA)
        return img
    

def create_video_with_counter(dir1: str, dir2: str, output_path: str,
                             start: int, end: int, repeat: int,
                             fps: float, title1: str, title2: str):
    # 参数类型验证
    if not all(os.path.isdir(d) for d in [dir1, dir2]):
        raise NotADirectoryError("输入目录不存在")
    
    # 获取图片尺寸（增加错误处理）
    sample_path = os.path.join(dir1, f"{start:03d}.png")
    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"起始帧不存在：{sample_path}")
        
    sample = cv2.imread(sample_path)
    if sample is None:
        raise ValueError("无法读取样本图片")
        
    h, w = sample.shape[:2]
    
    # 初始化计数器（显式类型声明）
    left_counter: GreenCounter = GreenCounter(title1, w, h)
    right_counter: GreenCounter = GreenCounter(title2, w, h)
    
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h))
    
    for idx in list(range(start, end+1)) * repeat:
        # 读取图片
        img1 = cv2.imread(os.path.join(dir1, f"{idx:03d}.png"))
        img2 = cv2.imread(os.path.join(dir2, f"{idx:03d}.png"))
        
        # 检测并计数
        left_counter.check_green(img1)
        right_counter.check_green(img2)
        
        # 添加标注
        img1 = left_counter.draw_info(img1)
        img2 = right_counter.draw_info(img2)
        
        # 拼接并写入
        combined = cv2.hconcat([img1, img2])
        video_writer.write(combined)
    
    video_writer.release()
    print(f"视频生成完成！最终计数 - 左: {left_counter.count}, 右: {right_counter.count}")

# 使用示例
if __name__ == "__main__":
    # 配置参数
    DIR1 = "./detection_results/before_attack"    # 第一个图片目录路径
    DIR2 = "./detection_results/after_attack"   # 第二个图片目录路径
    OUTPUT = "5703.mp4"      # 输出视频路径
    START = 180                      # 起始帧号
    END = 359                        # 结束帧号
    REPEAT = 2                       # 重复次数
    FPS = 45                         # 视频帧率
    TITLE_LEFT = "Vanilla 3DGS"    # 左侧标题
    TITLE_RIGHT = "3DGAA"   # 右侧标题

    # try:
    create_video_with_counter(
        dir1=DIR1,
        dir2=DIR2,
        output_path=OUTPUT,
        start=START,
        end=END,
        repeat=REPEAT,
        fps=FPS,
        title1=TITLE_LEFT,
        title2=TITLE_RIGHT
    )
    # except Exception as e:
    #     print(f"程序出错：{str(e)}")