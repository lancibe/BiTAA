#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_eval_absrel_rmse.py

Evaluate AbsRel and RMSE for one or many RGB images using a depth estimation model
(the same models you use in your experiments, e.g., MiDaS DPT-Large).

Supports two GT modes:
  (A) --gt_map PATH: dense ground-truth depth map (per-image or a single file)
  (B) --gt_scalar M  : a single known depth (in meters) for the ROI (e.g., object's true distance)

Metrics are computed over a mask region (ROI). Use one of:
  --mask PATH         : a binary mask image (255=valid), same size as the RGB
  --bbox x1 y1 x2 y2  : a bounding box ROI in pixels
  (if neither is given, the whole image is evaluated)

Scale/offset alignment options between the model's depth and GT:
  --align scale   : median-based scale alignment (default for MiDaS)
  --align affine  : linear fit y ≈ a*x + b on the ROI
  --align none    : no alignment

Outputs a CSV summary with per-image metrics and overall means.

Example:
  python depth_eval_absrel_rmse.py \
      --images "data/attacked/*.png" \
      --model midas_dpt_large \
      --gt_scalar 5.0 \
      --bbox 320 200 960 720 \
      --align scale \
      --out results_attacked.csv
"""
import argparse
import glob
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# ------------------------------
# Utilities
# ------------------------------
def read_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img  # HxWx3, BGR uint8

def read_mask(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    m = (m > 127).astype(np.uint8)
    return m  # HxW uint8 {0,1}

def bbox_to_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return m

def read_depth_map(path: str, shape_hw: Tuple[int, int]) -> np.ndarray:
    """Read a depth map as float32 (meters). Supports single-channel EXR/PNG/TIFF, or 16bit PNG scaled by --gt_scale16."""
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Cannot read depth map: {path}")
    if depth.ndim == 3:
        # If 3-channel (rare), take first channel
        depth = depth[..., 0]
    depth = depth.astype(np.float32)
    if depth.shape != shape_hw:
        depth = cv2.resize(depth, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return depth  # HxW float32 (assumed meters already; you can scale externally)

def median_scale_align(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    eps = 1e-6
    sel = (mask > 0) & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if sel.sum() < 10:
        return 1.0
    s = np.median(gt[sel] / (pred[sel] + eps))
    return float(s)

def affine_fit(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    """Return a,b s.t. gt ≈ a*pred + b on masked pixels (least squares)."""
    sel = (mask > 0) & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if sel.sum() < 10:
        return 1.0, 0.0
    x = pred[sel].reshape(-1, 1).astype(np.float64)
    y = gt[sel].reshape(-1, 1).astype(np.float64)
    # [x 1] theta = y  -> theta = (A^T A)^{-1} A^T y
    A = np.hstack([x, np.ones_like(x)])
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    a = float(theta[0, 0]); b = float(theta[1, 0])
    return a, b

def compute_absrel_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (AbsRel, RMSE, pred_median, gt_median) over mask. pred, gt: HxW float32 in meters."""
    eps = 1e-6
    sel = (mask > 0) & np.isfinite(pred) & np.isfinite(gt) & (gt > 0)
    if sel.sum() < 10:
        return np.nan, np.nan, np.nan, np.nan
    diff = pred[sel] - gt[sel]
    absrel = np.mean(np.abs(diff) / (gt[sel] + eps))
    rmse = float(np.sqrt(np.mean(diff**2)))
    return float(absrel), rmse, float(np.median(pred[sel])), float(np.median(gt[sel]))

# ------------------------------
# Depth models
# ------------------------------
class DepthModel:
    def __init__(self, name: str = "midas_dpt_large", device: Optional[str] = None):
        self.name = name.lower()
        self.device = torch.device(device) if device else (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = None
        self.transform = None
        self._init_model()

    def _init_model(self):
        if self.name in ["midas_dpt_large", "dpt_large", "midas"]:
            # MiDaS DPT-Large via torch.hub
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = transforms.dpt_transform
        elif self.name in ["midas_dpt_hybrid", "dpt_hybrid"]:
            self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid").to(self.device).eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = transforms.dpt_transform
        elif self.name in ["zoedepth", "zoe"]:
            try:
                from zoedepth.models.builder import build_model
                from zoedepth.utils.config import get_config
                conf = get_config("zoedepth", "infer")
                self.model = build_model(conf).to(self.device).eval()
                self.transform = None  # We'll use simple ImageNet normalization
            except Exception as e:
                print("[WARN] ZoeDepth not available; falling back to MiDaS DPT-Large.", file=sys.stderr)
                self.name = "midas_dpt_large"
                self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(self.device).eval()
                transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
                self.transform = transforms.dpt_transform
        else:
            raise ValueError(f"Unknown model: {self.name}")

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """Return HxW float32 depth (relative for MiDaS; meters-ish for ZoeDepth)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.name.startswith("midas"):
            net_in = self.transform(rgb).to(self.device)  # 1x3xhxw
            pred = self.model(net_in)
            pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=rgb.shape[:2], mode="bilinear", align_corners=False).squeeze(1)  # 1xHxW
            depth = pred[0].cpu().float().numpy()
            depth = depth.astype(np.float32)
            return depth
        elif self.name.startswith("zoe"):
            # Simple ImageNet norm
            t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0
            mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
            std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            t = (t - mean) / std
            t = t.unsqueeze(0).to(self.device)
            depth_m = self.model.infer(t)["metric_depth"].squeeze(0).cpu().float().numpy()
            return depth_m.astype(np.float32)

# ------------------------------
# Main eval
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True, help="Glob for images, e.g., 'data/*.png'")
    ap.add_argument("--model", type=str, default="midas_dpt_large", choices=["midas_dpt_large","midas_dpt_hybrid","zoedepth"], help="Depth estimator")
    ap.add_argument("--mask", type=str, default=None, help="Optional mask image path (255=valid). If set, used for all images.")
    ap.add_argument("--bbox", type=int, nargs=4, default=None, help="Optional bbox ROI: x1 y1 x2 y2")
    ap.add_argument("--gt_map", type=str, default=None, help="Optional GT depth map path or glob. If single file, used for all images.")
    ap.add_argument("--gt_scalar", type=float, default=None, help="Optional scalar GT distance (meters) for ROI (assume approx constant depth in ROI).")
    ap.add_argument("--align", type=str, default="scale", choices=["scale","affine","none"], help="Alignment mode between pred and GT")
    ap.add_argument("--out", type=str, default="depth_eval_results.csv", help="CSV output path")
    ap.add_argument("--fixed_scale_csv", type=str, default=None, help="CSV from a clean run (with --align scale). Apply its per-image scale_s to this run.")

    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if len(paths) == 0:
        print(f"No images match: {args.images}")
        sys.exit(1)

    # Prepare GT map paths if provided
    gt_map_paths = None
    if args.gt_map:
        g = sorted(glob.glob(args.gt_map))
        if len(g) == 0 and os.path.isfile(args.gt_map):
            gt_map_paths = [args.gt_map] * len(paths)
        elif len(g) == 1 and len(paths) > 1:
            gt_map_paths = g * len(paths)
        elif len(g) == len(paths):
            gt_map_paths = g
        else:
            print("[WARN] gt_map glob count does not match images; will try to reuse single map if one found.")
            gt_map_paths = (g * ((len(paths)+len(g)-1)//len(g)))[:len(paths)] if len(g)>0 else None

    # Load fixed scales from a clean run (optional)
    fixed_scale_map = None
    if args.fixed_scale_csv:
        import csv
        fixed_scale_map = {}
        with open(args.fixed_scale_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if "image" in row and "scale_s" in row and len(row["scale_s"]) > 0:
                    # key by basename to be robust
                    fixed_scale_map[os.path.basename(row["image"])] = float(row["scale_s"])
        print(f"[INFO] Loaded {len(fixed_scale_map)} fixed scales from {args.fixed_scale_csv}")


    # Model
    depth_model = DepthModel(args.model)

    # CSV header
    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["image","absrel","rmse","pred_median","gt_median","scale_s","affine_a","affine_b","num_pixels"])

    absrels = []; rmses = []

    for i, path in enumerate(paths):
        bgr = read_image_bgr(path)
        H, W = bgr.shape[:2]

        # ROI mask
        if args.mask:
            mask = read_mask(args.mask, (H,W))
        elif args.bbox is not None:
            x1,y1,x2,y2 = args.bbox
            mask = bbox_to_mask(H,W,x1,y1,x2,y2)
        else:
            mask = np.ones((H,W), dtype=np.uint8)

        # Predict depth
        pred = depth_model.predict(bgr)  # HxW float32
        assert pred.shape[:2] == (H,W), "Prediction size mismatch"

        # Ground truth
        if gt_map_paths is not None:
            gt = read_depth_map(gt_map_paths[i], (H,W))
        elif args.gt_scalar is not None:
            gt = np.full((H,W), float(args.gt_scalar), dtype=np.float32)
        else:
            print(f"[{i+1}/{len(paths)}] {os.path.basename(path)}: No GT provided (gt_map or gt_scalar). Skipping.")
            continue

        # Alignment
        s = 1.0; a = 1.0; b = 0.0
        pred_aligned = pred.copy()

        # ---- Fixed-scale has highest priority ----
        if fixed_scale_map is not None:
            key = os.path.basename(path)
            if key in fixed_scale_map:
                s = fixed_scale_map[key]
                pred_aligned = pred * s
            else:
                print(f"[WARN] No fixed scale for {key}; falling back to --align {args.align}")

        # ---- If no fixed scale, fall back to requested alignment ----
        if fixed_scale_map is None:
            if args.align == "scale":
                s = median_scale_align(pred, gt, mask)
                pred_aligned = pred * s
            elif args.align == "affine":
                a,b = affine_fit(pred, gt, mask)
                pred_aligned = a * pred + b
            # elif 'none': keep pred_aligned as-is


        # Metrics
        absrel, rmse, pm, gm = compute_absrel_rmse(pred_aligned, gt, mask)
        npx = int((mask>0).sum())
        absrels.append(absrel); rmses.append(rmse)

        print(f"[{i+1}/{len(paths)}] {os.path.basename(path)} | AbsRel={absrel:.4f} | RMSE={rmse:.4f} | pred_med={pm:.3f} | gt_med={gm:.3f} | s={s:.3f} | a={a:.3f} | b={b:.3f} | N={npx}")
        writer.writerow([path, f"{absrel:.6f}", f"{rmse:.6f}", f"{pm:.6f}", f"{gm:.6f}", f"{s:.6f}", f"{a:.6f}", f"{b:.6f}", npx])

    # Summary
    if len(absrels) > 0:
        mean_absrel = float(np.nanmean(absrels))
        mean_rmse = float(np.nanmean(rmses))
        print(f"\n== Summary over {len(absrels)} images ==")
        print(f"Mean AbsRel = {mean_absrel:.4f}")
        print(f"Mean RMSE   = {mean_rmse:.4f}")
        writer.writerow([])
        writer.writerow(["MEAN", f"{mean_absrel:.6f}", f"{mean_rmse:.6f}", "", "", "", "", "", ""])

    fout.close()

if __name__ == "__main__":
    main()
