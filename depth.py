import sys
import os, shutil
import torch
import torch.nn.functional as F

class Mono2Wrapper(torch.nn.Module):
    def __init__(self, repo_path, weights_dir, device, num_layers=18):
        super().__init__()
        assert os.path.isdir(repo_path), f"monodepth2 repo not found: {repo_path}"
        assert os.path.isdir(weights_dir), f"weights dir not found: {weights_dir}"
        sys.path.insert(0, repo_path)
        from networks.depth_decoder import DepthDecoder
        from networks.resnet_encoder import ResnetEncoder

        enc_path  = os.path.join(weights_dir, "encoder.pth")
        dec_path  = os.path.join(weights_dir, "depth.pth")
        enc_state = torch.load(enc_path, map_location="cpu")
        feed_w, feed_h = enc_state.get('height', 192), enc_state.get('width', 640)

        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=True)
        # strip 'module.' if present
        enc_state = {k.replace("module.", ""): v for k,v in enc_state.items() if k in self.encoder.state_dict() or k.replace("module.","") in self.encoder.state_dict()}
        self.encoder.load_state_dict(enc_state, strict=False)

        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        dec_state = torch.load(dec_path, map_location="cpu")
        dec_state = {k.replace("module.", ""): v for k,v in dec_state.items()}
        self.decoder.load_state_dict(dec_state, strict=False)

        self.feed_h, self.feed_w = feed_h, feed_w  # 注意：README 的分辨率是 640x192（W×H）
        self.to(device).eval()

    @torch.no_grad()
    def forward_disp(self, img_chw01: torch.Tensor) -> torch.Tensor:
        # 推理版（不反传）
        C,H,W = img_chw01.shape
        net_in = F.interpolate(img_chw01.unsqueeze(0), size=(self.feed_h, self.feed_w), mode="bilinear", align_corners=False)
        # 按官方代码，通常就是 (0,1) 归一输入；若有需要可加 ImageNet normalize
        feats = self.encoder(net_in)
        out   = self.decoder(feats)[("disp", 0)]  # (1,1,Hf,Wf)
        disp  = F.interpolate(out, size=(H,W), mode="bilinear", align_corners=False).squeeze(0)
        return disp.clamp_min(1e-6)

    def forward_disp_grad(self, img_chw01: torch.Tensor) -> torch.Tensor:
        # 训练版（支持反传）
        C,H,W = img_chw01.shape
        net_in = F.interpolate(img_chw01.unsqueeze(0), size=(self.feed_h, self.feed_w), mode="bilinear", align_corners=False)
        feats = self.encoder(net_in)
        out   = self.decoder(feats)[("disp", 0)]
        disp  = F.interpolate(out, size=(H,W), mode="bilinear", align_corners=False).squeeze(0)
        return disp.clamp_min(1e-6)


class DepthEstimator:
    """
    统一接口：predict / predict_nograd
    backend: "dpt" | "monodepth2" | "depthanything"
    对 monodepth2:
      - mono_source: "hub" | "local" | "weights"
      - mono_repo_path: 本地 monodepth2 仓库路径 (source="local")
      - mono_variant: "mono_640x192" / "mono+stereo_640x192" / "mono_no_pt_640x192" ...
      - mono_weights_dir: 本地权重目录 (source="weights")，需包含 encoder.pth / depth.pth
    """
    def __init__(self,
                 device,
                 backend="dpt",
                 # --- 通用 ---
                 net_size=None,
                 amp=True,
                 amp_dtype=torch.float16,
                 # --- monodepth2 选项 ---
                 mono_source="hub",
                 mono_repo_path=None,
                 mono_variant="mono_640x192",
                 mono_weights_dir=None,
                 force_reload_hub=False,
                 # --- depthanything 选项 ---
                 depthanything_ckpt="LiheYoung/depth-anything-base-hf",
                 ):
        self.device = device
        self.backend = backend.lower()
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.eps = 1e-6

        if self.backend == "dpt":
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device).eval()
            for p in self.model.parameters(): p.requires_grad_(False)
            self.net_size = net_size or (384, 384)
            self.normalize = lambda x: (x - 0.5) / 0.5

        elif self.backend == "depthanything":
            from transformers import AutoModelForDepthEstimation
            self.model = AutoModelForDepthEstimation.from_pretrained(depthanything_ckpt).to(device).eval()
            for p in self.model.parameters(): p.requires_grad_(False)
            self.net_size = net_size or (384, 384)  # 518→384 更省显存
            self.normalize = lambda x: (x - 0.5) / 0.5

        elif self.backend == "monodepth2":
            self.model = Mono2Wrapper(mono_repo_path, mono_weights_dir, device)
            self.min_depth, self.max_depth = 0.1, 100.0
            for p in self.model.parameters(): p.requires_grad_(False)

    # ---------- Monodepth2: hub 方式，自动清坏缓存 ----------
    def _load_mono_from_hub(self, force_reload=False):
        def try_load(fr=False):
            return torch.hub.load("nianticlabs/monodepth2", self.mono_variant, pretrained=True, force_reload=fr)
        try:
            return try_load(fr=force_reload)
        except FileNotFoundError:
            hub_dir = torch.hub.get_dir()
            for d in ("nianticlabs_monodepth2_master", "nianticlabs_monodepth2_main"):
                p = os.path.join(hub_dir, d)
                if os.path.isdir(p): shutil.rmtree(p)
            return try_load(fr=True)

    # ---------- Monodepth2: 纯本地构建 ----------
    def _build_mono_from_weights(self, weights_dir):
        """
        需要两份权重：
          encoder.pth  (ResNet 编码器)
          depth.pth    (DepthDecoder)
        你可以从作者发布的模型包中分离出来保存为这两个文件。
        """
        import sys
        # 假设你已经 `git clone https://github.com/nianticlabs/monodepth2` 到某处
        # 并将该目录加入 sys.path
        # 例如：sys.path.insert(0, "/path/to/monodepth2")
        try:
            import networks  # 来自 monodepth2 仓库
        except Exception as e:
            raise ImportError("请先 git clone monodepth2 并将其目录加入 sys.path，再使用 mono_source='weights'") from e

        encoder = networks.ResnetEncoder(18, False)
        depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        enc_path = os.path.join(weights_dir, "encoder.pth")
        dec_path = os.path.join(weights_dir, "depth.pth")
        assert os.path.isfile(enc_path) and os.path.isfile(dec_path), "在 mono_weights_dir 中未找到 encoder.pth / depth.pth"

        loaded_dict_enc = torch.load(enc_path, map_location="cpu")
        # 对齐 state_dict 键
        encoder.load_state_dict({k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()})
        depth_decoder.load_state_dict(torch.load(dec_path, map_location="cpu"))
        model = torch.nn.Module()
        model.encoder = encoder
        model.depth_decoder = depth_decoder

        # 统一 forward：输出 disparity 张量，形状 [B,1,h,w]
        def fwd(x):
            feats = model.encoder(x)
            out = model.depth_decoder(feats)
            disp = out[("disp", 0)] if isinstance(out, dict) and ("disp", 0) in out else out
            if disp.dim() == 3: disp = disp.unsqueeze(1)
            return disp
        model.forward = fwd
        return model
    
    def _disp_to_depth(self, disp: torch.Tensor,
                   min_depth: float = None,
                   max_depth: float = None) -> torch.Tensor:
        """
        官方映射：min/max depth -> min/max disparity -> depth = 1/scaled_disp
        输入 disp 形状 [1,H,W] 或 [B,1,H,W]；返回 [1,H,W] 或 [B,1,H,W]
        """
        md = self.min_depth if min_depth is None else min_depth
        Md = self.max_depth if max_depth is None else max_depth

        if disp.dim() == 3:     # [1,H,W] -> [1,1,H,W]
            disp = disp.unsqueeze(0)
        if disp.dim() == 4 and disp.size(1) != 1:
            # 若是 [B,H,W]，补成 [B,1,H,W]
            disp = disp.unsqueeze(1)

        min_disp, max_disp = 1.0 / Md, 1.0 / md
        scaled_disp = min_disp + (max_disp - min_disp) * disp.clamp(0, 1)
        depth = 1.0 / (scaled_disp + self.eps)
        return depth.squeeze(0) if depth.size(0) == 1 else depth


    # ---------- 公共工具 ----------
    def _prep(self, img_chw01):
        if img_chw01.size(0) == 1: img_chw01 = img_chw01.repeat(3,1,1)
        h,w = self.net_size
        x = F.interpolate(img_chw01.unsqueeze(0), size=(h,w), mode="bilinear", align_corners=False)
        x = self.normalize(x)
        return x.to(self.device).contiguous(memory_format=torch.channels_last)

    def _post(self, pred, H, W):
        if pred.dim()==3: pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(H,W), mode="bilinear", align_corners=False)
        return pred.squeeze(0)

    # ---------- 三后端前向 ----------
    def _forward_dpt(self, x):  # -> depth-like [1,1,h,w]
        y = self.model(x)
        return y.unsqueeze(1) if y.dim()==3 else y

    def _forward_depthanything(self, x):
        from contextlib import nullcontext
        ctx = torch.cuda.amp.autocast(dtype=self.amp_dtype) if self.amp else nullcontext()
        with ctx:
            outputs = self.model(pixel_values=x)
        y = None
        # 1) dict 风格
        if isinstance(outputs, dict):
            for key in ("predicted_depth", "depth", "logits", "last_hidden_state"):
                if key in outputs and outputs[key] is not None:
                    y = outputs[key]
                    break
        else:
            # 2) 带属性的 ModelOutput 风格
            for attr in ("predicted_depth", "depth", "logits", "last_hidden_state"):
                if hasattr(outputs, attr):
                    val = getattr(outputs, attr)
                    if val is not None:
                        y = val
                        break

        if y is None:
            raise RuntimeError("DepthAnything output does not contain a depth tensor (checked predicted_depth/depth/logits/last_hidden_state).")
        
        if y.dim() == 3:  # [B,H,W] -> [B,1,H,W]
            y = y.unsqueeze(1)
        return y

    def _forward_monodepth2(self, x):
        out = self.model(x)
        if isinstance(out, dict) and ("disp",0) in out:
            disp = out[("disp",0)]
        else:
            disp = out
        if disp.dim()==3: disp = disp.unsqueeze(1)
        min_depth, max_depth = 0.1, 100.0  # 与官方一致，可做超参
        min_disp, max_disp = 1.0/max_depth, 1.0/min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp.clamp(0,1)
        depth = 1.0 / (scaled_disp + self.eps)
        return depth

    # ---------- API ----------
    @torch.no_grad()
    def predict_nograd(self, img_chw01: torch.Tensor) -> torch.Tensor:
        C,H,W = img_chw01.shape
        if self.backend == "monodepth2":
            # FIX: 用官方 disp_to_depth，而不是 1/disp
            disp = self.model.forward_disp(img_chw01)        # [1,H,W]
            depth = self._disp_to_depth(disp)                # <--- 用官方映射
            return depth.clamp_min(self.eps).detach()
        elif self.backend == "dpt":
            net_in = self._prep(img_chw01)
            pred = self._forward_dpt(net_in)
            return self._post(pred, H, W).clamp_min(self.eps).detach()
        else:  # depthanything
            net_in = self._prep(img_chw01)                   # FIX: 走 HF 分支
            pred = self._forward_depthanything(net_in)
            return self._post(pred, H, W).clamp_min(self.eps).detach()

    def predict(self, img_chw01: torch.Tensor) -> torch.Tensor:
        C,H,W = img_chw01.shape
        if self.backend == "monodepth2":
            # FIX: 同样用 disp_to_depth，保持与 no-grad 一致
            disp = self.model.forward_disp_grad(img_chw01)   # [1,H,W]
            depth = self._disp_to_depth(disp)                # <--- 用官方映射
            return depth.clamp_min(self.eps)
        elif self.backend == "dpt":
            net_in = self._prep(img_chw01)
            pred = self._forward_dpt(net_in)
            return self._post(pred, H, W).clamp_min(self.eps)
        else:  # depthanything
            net_in = self._prep(img_chw01)
            pred = self._forward_depthanything(net_in)
            return self._post(pred, H, W).clamp_min(self.eps)