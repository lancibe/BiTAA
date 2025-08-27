# 由于FRCNN, MRCNN, SSD都可以方便的使用torchvision调用, 这里单独实现YOLOv3的目标检测模型

from ultralytics import YOLO
import torch
import contextlib
import io

class YOLOv3Detector:
    def __init__(self, device):
        with contextlib.redirect_stdout(io.StringIO()):
            self.detector = YOLO("yolov3.pt").to(device)
        self.device = device

    def forward(self, image):
        image = image.to(self.device)
        results = self.detector(image)
        # print(results)
        # 将 YOLO 输出的结果转换为类似字典的结构，方便调用
        outputs = []
        for result in results:
            # 从 result 中提取类别标签和边界框
            boxes = result.boxes.xyxy  # 获取边界框
            labels = result.boxes.cls  # 获取类别标签
            scores = result.boxes.conf  # 获取每个预测的置信度

            # 创建和 Faster R-CNN 类似的字典结构
            output = {
                'boxes': boxes,
                'labels': labels.to(torch.int64),  # 将标签转换为 int64 格式
                'scores': scores
            }
            outputs.append(output)
        
        return outputs