#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小可复现、稳定版 FLQ-YOLOv（只量化梯度，不量化模型本体）

核心策略：
  - 保持全局模型 FP32（绝对不能量化）
  - 上行梯度向量量化（1bit / 8bit / 32bit）
  - 下行模型保持 FP32
  - 确保 YOLO 在 Round>1 时不会崩溃
"""

import torch
from ultralytics import YOLO
import yaml
from pathlib import Path
import argparse
import csv
import matplotlib.pyplot as plt

# ====================== 工具函数 ==========================

def sd_to_vec(sd):
    return torch.cat([p.view(-1) for p in sd.values()])

def vec_to_sd(vec, tmpl):
    out = {}
    idx = 0
    for k, v in tmpl.items():
        n = v.numel()
        out[k] = vec[idx:idx+n].view_as(v)
        idx += n
    return out

def quantize(v, bits):
    if bits == 32:
        return v, 1.0, 0.0
    v = v.detach()
    mn, mx = v.min(), v.max()
    if bits == 1:
        scale = max(abs(mn), abs(mx))
        q = torch.sign(v)
        return q, scale, 0.0
    levels = 2 ** bits
    scale = (mx - mn) / (levels - 1)
    q = torch.round((v - mn) / scale).clamp(0, levels-1)
    return q, scale, mn

def dequantize(q, scale, zp, bits):
    if bits == 32:
        return q
    if bits == 1:
        return q * scale
    return q * scale + zp

# ====================== 主流程 =============================

def run(data_yaml, model_path, out_dir, rounds=5, epochs=1, batch=8, bits=1):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取类别数
    with open(data_yaml, "r") as f:
        nc = yaml.safe_load(f)["nc"]

    # 初始化 YOLO
    model = YOLO(model_path)
    from ultralytics.nn.tasks import DetectionModel
    model.model = DetectionModel(model.model.yaml, ch=3, nc=nc)

    # 初始全局模型(FP32)
    global_sd = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}

    rows = []
    bits_up_cum = 0
    bits_down_cum = 0

    for r in range(rounds):
        print(f"\n===== Round {r} =====")

        # -------- 下行（不量化模型）---------
        model.model.load_state_dict(global_sd, strict=False)

        # 模型大小（FP32）
        vec = sd_to_vec(global_sd)
        bits_down = vec.numel() * 32
        bits_down_cum += bits_down

        # -------- 本地训练 --------
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=640,
            device='cuda',
            workers=0,
            project=str(out_dir),
            name=f"round_{r}",
            exist_ok=True,
            verbose=False,
            val=True,
        )

        # 提取指标（YOLO 官方 results.csv 最靠谱）
        csv_file = out_dir / f"round_{r}" / "results.csv"
        if csv_file.exists():
            with open(csv_file) as f:
                last = list(csv.DictReader(f))[-1]
            map50 = float(last.get("metrics/mAP50(B)", last.get("metrics/mAP50", 0.0)))
            map95 = float(last.get("metrics/mAP50-95(B)", last.get("metrics/mAP50-95", 0.0)))
        else:
            map50 = 0.0
            map95 = 0.0

        print(f"mAP50={map50:.4f}, mAP50-95={map95:.4f}")

        # -------- 计算更新向量（FP32梯度） --------
        local_sd = {k: v.cpu().clone() for k, v in model.model.state_dict().items()}
        grad_vec = sd_to_vec(local_sd) - sd_to_vec(global_sd)

        # -------- 上行梯度量化 --------
        q, s, zp = quantize(grad_vec, bits)
        bits_up = grad_vec.numel() * bits
        bits_up_cum += bits_up

        # -------- 服务器更新全局模型 (FP32) --------
        updated_vec = dequantize(q, s, zp, bits)
        new_vec = sd_to_vec(global_sd) + updated_vec
        global_sd = vec_to_sd(new_vec, global_sd)

        # -------- 记录 --------
        rows.append(dict(
            round=r,
            map50=map50,
            map50_95=map95,
            bits_up=bits_up,
            bits_down=bits_down,
            bits_up_cum=bits_up_cum,
            bits_down_cum=bits_down_cum
        ))

    # 保存CSV
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\n指标写入 {csv_path}")

    # -------- 绘图 --------
    plt.figure()
    plt.plot([r["round"] for r in rows], [r["map50"] for r in rows])
    plt.xlabel("Round"); plt.ylabel("mAP50"); plt.grid()
    plt.savefig(out_dir / "map50.png")

    print("完成！")


# ====================== CLI ===============================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--out", type=str, default="flq_minimal_out")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--bits", type=int, default=1)
    args = p.parse_args()

    run(
        data_yaml=args.data,
        model_path=args.model,
        out_dir=args.out,
        rounds=args.rounds,
        epochs=args.epochs,
        batch=args.batch,
        bits=args.bits,
    )
