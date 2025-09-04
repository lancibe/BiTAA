import torch
import torch.nn.functional as F

class DepthEstimator:
    """
    统一接口：
      .predict(img_chw01: FloatTensor[C,H,W] in [0,1]) -> depth_chw: FloatTensor[1,H,W] (meters or pseudo-meters)
      .predict_nograd(...) 同上，但禁用梯度
    切换后端：将 MODEL 改成 "dpt" / "monodepth2" / "depthanything"
    """
    # >>> 选择后端： "dpt" | "monodepth2" | "depthanything"
    # MODEL = "monodepth2"
    MODEL = "dpt"  # 默认 DPT-Large；改为 "monodepth2" 或 "depthanything" 即可
    # MODEL = "depthanything"
    

    def __init__(self, device: torch.device):
        self.device = device
        self.model_type = self.MODEL.lower()
        self.eps = 1e-6

        if self.model_type == "dpt":
            # DPT-Large (MiDaS) via torch.hub (冻结参数)
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.net_size = (384, 384)
            # 简洁归一化（与您原实现一致；如需更高精度可替换为官方 MiDaS transforms 的纯 Torch 版本）
            self.normalize = lambda x: (x - 0.5) / 0.5

        elif self.model_type == "monodepth2":
            # Monodepth2 via torch.hub; 模型名可根据需要改成 "mono_640x192" / "mono+stereo_640x192" 等
            self.model = torch.hub.load("nianticlabs/monodepth2", "mono_640x192", pretrained=True).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.net_size = (192, 640)  # (H,W)
            # Monodepth2 输入一般为 [0,1] 归一化即可
            self.normalize = lambda x: x

        elif self.model_type == "depthanything":
            # Depth Anything V1 via Hugging Face Transformers
            try:
                from transformers import AutoModelForDepthEstimation
            except Exception as e:
                raise ImportError("需要安装 transformers>=4.36：pip install transformers") from e
            # 模型名可按需替换为相应 checkpoint（例如 'LiheYoung/depth-anything-large-hf'）
            self.model = AutoModelForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-large-hf"
            ).to(device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
            # DepthAnything V1 常用 518×518；如显存紧张可改小
            self.net_size = (518, 518)
            # 使用对称归一化；如需严格对齐官方，可替换为 ImageNet mean/std
            self.normalize = lambda x: (x - 0.5) / 0.5
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    # -------- 内部公共工具 --------
    def _prep(self, img_chw01: torch.Tensor) -> torch.Tensor:
        """
        纯 Torch 预处理：resize 到网络尺寸 + 归一化；保留梯度
        输入：img_chw01 [3,H,W], 值域[0,1]
        输出：net_in [1,3,h,w]
        """
        assert img_chw01.dim() == 3 and img_chw01.size(0) in (1,3), "expect CHW image"
        # 若是单通道，重复到 3 通道
        if img_chw01.size(0) == 1:
            img_chw01 = img_chw01.repeat(3, 1, 1)
        h, w = self.net_size
        x = F.interpolate(img_chw01.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
        x = self.normalize(x)
        return x

    def _post_resize(self, pred: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        将网络输出 resize 回原图尺寸；输出形状 [1,H,W]
        """
        if pred.dim() == 3:  # [1,h,w]
            pred = pred.unsqueeze(1)  # -> [1,1,h,w]
        elif pred.dim() == 2:  # [h,w]
            pred = pred.unsqueeze(0).unsqueeze(0)
        pred = F.interpolate(pred, size=(H, W), mode="bilinear", align_corners=False)
        return pred.squeeze(0)  # [1,H,W]

    # -------- 三个后端的前向封装（支持梯度）--------
    def _forward_dpt(self, net_in: torch.Tensor) -> torch.Tensor:
        # DPT 输出可能为 [1,1,h,w] 或 [1,h,w]
        out = self.model(net_in)
        if out.dim() == 3:
            out = out.unsqueeze(1)
        # DPT 输出为“相对深度”，直接当做伪米制使用
        return out  # [1,1,h,w]

    def _forward_monodepth2(self, net_in: torch.Tensor) -> torch.Tensor:
        # Monodepth2 前向：返回 dict，其中 'disp' 为视差；不同 hub 版本可能略有差异
        out = self.model(net_in)[("disp", 0)] if isinstance(self.model(net_in), dict) else self.model(net_in)
        if out.dim() == 3:
            out = out.unsqueeze(1)  # [1,1,h,w]
        # 视差 -> 伪深度（避免除零）
        depth = 1.0 / (out + self.eps)
        return depth

    def _forward_depthanything(self, net_in: torch.Tensor) -> torch.Tensor:
        # Transformers 模型通常期望键 'pixel_values'；我们直接传 tensor 做为 input（B,C,H,W）
        outputs = self.model(pixel_values=net_in)
        if hasattr(outputs, "predicted_depth"):
            out = outputs.predicted_depth  # [B,1,h,w]
        else:
            # 兜底：某些实现将深度放在 logits 上
            out = outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out

    # -------- 公共 API --------
    @torch.no_grad()
    def predict_nograd(self, img_chw01: torch.Tensor) -> torch.Tensor:
        C, H, W = img_chw01.shape
        net_in = self._prep(img_chw01.to(self.device))
        if self.model_type == "dpt":
            pred = self._forward_dpt(net_in)
        elif self.model_type == "monodepth2":
            pred = self._forward_monodepth2(net_in)
        else:  # depthanything
            pred = self._forward_depthanything(net_in)
        pred = self._post_resize(pred, H, W)
        return pred.clamp_min(self.eps).detach()

    def predict(self, img_chw01: torch.Tensor) -> torch.Tensor:
        """
        允许梯度回传至输入（及你的可微渲染器），任务模型本身参数冻结。
        """
        C, H, W = img_chw01.shape
        net_in = self._prep(img_chw01.to(self.device))
        if self.model_type == "dpt":
            pred = self._forward_dpt(net_in)
        elif self.model_type == "monodepth2":
            pred = self._forward_monodepth2(net_in)
        else:  # depthanything
            pred = self._forward_depthanything(net_in)
        pred = self._post_resize(pred, H, W)
        return pred.clamp_min(self.eps)