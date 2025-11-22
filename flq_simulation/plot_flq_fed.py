# -*- coding: utf-8 -*-
import argparse, os, numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt

# ---- 中文字体与编码：避免乱码、负号方块 ----
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
                                          'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 模式别名
MODE_ALIASES = {
    "qgd": "qgd", "fedavg": "fedavg",
    "flq_lowbit": "bbit", "bbit": "bbit",
    "flq_bin": "bin", "bin": "bin",
    "laq8": "laq8"
}
MODES_DEFAULT = ["qgd", "bbit", "laq8"]

# ---- 平滑工具 ----
def moving_avg(y: np.ndarray, win: int) -> np.ndarray:
    win = int(max(1, win))
    if win == 1: return y
    pad = win // 2
    ypad = np.pad(y, (pad, win - 1 - pad), mode='edge')
    ker = np.ones(win, dtype=np.float64) / float(win)
    return np.convolve(ypad, ker, mode='valid')

def smooth_series(y: np.ndarray, win: int, ema: float = 0.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if win and win > 1:
        y = moving_avg(y, win)
    if ema and 0.0 < ema < 1.0:
        out = np.empty_like(y)
        out[0] = y[0]
        a = ema
        for i in range(1, len(y)):
            out[i] = a * out[i-1] + (1 - a) * y[i]
        y = out
    return y.astype(np.float64)

# ---- 可选图（保留）----
def draw_fig1(excel_map, use_bits=False, which_bits="cum_bits_total",
              y="loss", title="收敛对比"):
    fig, ax = plt.subplots()
    for name, path in excel_map.items():
        df = pd.read_excel(path, sheet_name=f"curve_{name}")
        xs = df[which_bits] if use_bits else df["iter"]
        ax.plot(xs, df[y], label=name.upper(), linewidth=2.0)
    ax.set_xlabel("累计通信比特" if use_bits else "迭代轮次", fontweight='bold')
    ax.set_ylabel("损失" if y == "loss" else y, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    leg = ax.legend(frameon=True)
    leg.set_draggable(True)
    for txt in leg.get_texts(): txt.set_fontweight('bold')
    ax.tick_params(labelsize=12, width=1.5)
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
    fig.tight_layout()

def draw_binary_series(excel_path, mode):
    try:
        df = pd.read_excel(excel_path, sheet_name=f"bin_{mode}")
    except Exception:
        return
    fig, ax = plt.subplots()
    ax.scatter(df["comm"], df["bit"], s=10)
    ax.axhline(0.5, linewidth=1.0)
    ax.set_xlabel("通信轮次", fontweight='bold')
    ax.set_ylabel("二值符号", fontweight='bold')
    ax.set_title("通信中的二值量化结果", fontweight='bold')
    ax.tick_params(labelsize=12, width=1.5)
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
    fig.tight_layout()

# ===== 论文 Fig.2：迭代轮次 vs 交叉熵（QGD / LAQ / FLQ） =====
def draw_paper_fig2_iter(excel_map, max_iter=800, smooth_win=9, smooth_ema=0.0,
                         legend_size=14, line_width=2.5, title=None):
    # 选择 FLQ 曲线：优先 bbit，无则退回 bin
    flq_key = "bbit" if "bbit" in excel_map else ("bin" if "bin" in excel_map else None)
    order = [("qgd", "QGD", "-"),
             ("fedavg", "FedAvg", "-"),
             ("laq8",   "LAQ",  "--")]
    if flq_key:
        order.append((flq_key, "FLQ（本算法）", "-"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for key, label, ls in order:
        if key not in excel_map: 
            continue
        
        # 智能查找 sheet 名称
        xl = pd.ExcelFile(excel_map[key])
        sheet_name = f"curve_{key}"
        if sheet_name not in xl.sheet_names:
            # 尝试找任何以 curve_ 开头的 sheet
            candidates = [s for s in xl.sheet_names if s.startswith("curve_")]
            if candidates:
                sheet_name = candidates[0]
                print(f"Warning: sheet 'curve_{key}' not found in {excel_map[key]}, using '{sheet_name}' instead.")
            else:
                print(f"Error: No curve sheet found in {excel_map[key]}")
                continue

        df = pd.read_excel(excel_map[key], sheet_name=sheet_name)
        df = df.iloc[:max_iter]
        ycol = "entropy" if "entropy" in df.columns else "loss"
        y = df[ycol].to_numpy()
        y = smooth_series(y, win=smooth_win, ema=smooth_ema)
        x = np.arange(1, len(y)+1, dtype=np.int32)  # 平滑后对齐到新长度
        ax.plot(x, y, linestyle=ls, linewidth=line_width, label=label)

    ax.set_xlabel("迭代轮次", fontweight='bold')
    ax.set_ylabel("交叉熵损失", fontweight='bold')
    if title: ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, len(x) if 'x' in locals() else max_iter)
    leg = ax.legend(fontsize=legend_size, frameon=True)
    for txt in leg.get_texts(): txt.set_fontweight('bold')
    ax.tick_params(labelsize=12, width=1.8)
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
    fig.tight_layout()
    return fig, ax

# ===== 论文 Fig.3：通信中的二值量化结果 =====
# def draw_paper_fig3_binary(excel_path: str, mode_key="bin", max_comm=200):
#     # 读取训练时记录的比特流（需要 v3 日志）
#     df = pd.read_excel(excel_path, sheet_name=f"bin_{mode_key}")
#     if max_comm is not None:
#         df = df.iloc[:max_comm]
#     x = df["comm"].to_numpy()
#     y = df["bit"].to_numpy()

#     fig, ax = plt.subplots(figsize=(7.2, 4.0))
#     ax.scatter(x, y, s=12, marker='o')               # 蓝点
#     ax.hlines(0.5, x.min(), x.max(), colors='r', linewidth=1.8)  # 0.5 参考线
#     ax.set_xlabel("通信轮次", fontweight='bold')
#     ax.set_ylabel("二值取值", fontweight='bold')
#     ax.set_ylim(-0.05, 1.05)
#     ax.set_xlim(x.min(), x.max())
#     ax.tick_params(labelsize=12, width=1.8)
#     for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
#     fig.tight_layout()
#     return fig, ax

def draw_paper_fig3_binary(excel_path: str, mode_key="bin", max_comm=200):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt
    df = pd.read_excel(excel_path, sheet_name=f"bin_{mode_key}")
    if max_comm is not None:
        df = df.iloc[:max_comm]
    x = df["comm"].to_numpy()
    y = df["bit"].to_numpy().astype(np.float32)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # 先画所有列的 0→0.5 浅灰梳状线
    ax.vlines(x, 0.0, 0.5, colors='0.85', linewidth=1.0, zorder=1)
    # 对 bit==1 的列，补画 0.5→1.0 深灰梳状线
    idx1 = (y >= 0.5)
    ax.vlines(x[idx1], 0.5, 1.0, colors='0.6', linewidth=1.2, zorder=2)
    # 顶部散点
    ax.scatter(x, y, s=16, zorder=3)
    # 中位线
    ax.hlines(0.5, x.min(), x.max(), colors='r', linewidth=2.0, zorder=0)

    ax.set_xlabel("通信轮次", fontweight='bold')
    ax.set_ylabel("二值取值", fontweight='bold')
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(x.min(), x.max())
    ax.tick_params(labelsize=12, width=1.8)
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
    fig.tight_layout()
    return fig, ax

# ===== 论文 Fig.4：不同 k 的量化逼近（受控半径 + 高斯真值） =====
import numpy as np
import matplotlib.pyplot as plt

def _rand_unit(d, rng):
    v = rng.standard_normal(d).astype(np.float32)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _orth_unit_to(x, rng):
    d = x.size
    v = rng.standard_normal(d).astype(np.float32)
    # 去掉与 x 的分量，做正交单位向量
    v = v - np.dot(v, x) * x
    n = np.linalg.norm(v) + 1e-12
    return v / n

def _quant_vec_kbit_stoch(v, k, rng):
    # 逐张量=k=向量级：与训练里逐张量缩放一致，这里只有一个张量
    L = float(2**(k-1) - 1)
    amax = float(np.max(np.abs(v))) + 1e-12
    s = amax / L
    y = v / s
    low = np.floor(y); p = y - low
    rnd = (rng.random(y.shape) < p).astype(np.float32)
    q = np.clip(low + rnd, -L, L) * s
    return q.astype(np.float32)

def draw_paper_fig4_quant_controlled(ks=(4,8), d=64, n=3000, r_max=5.0, sigma=2.0, seed=42):
    """
    受控半径 r∈[0,r_max]；真值 g(r)=exp(-(r/sigma)^2)；对 x,y 做 k-bit 量化后计算近似。
    输出与论文风格一致的两子图，并打印 RMSE。
    """
    rng = np.random.default_rng(seed)
    # 生成一组 (r, true) 样本
    rs = rng.random(n).astype(np.float32) * float(r_max)   # 均匀覆盖 [0, r_max]
    true = np.exp(- (rs / float(sigma))**2 ).astype(np.float32)

    figs = []
    for k in ks:
        approx = np.empty(n, np.float32)
        for i, r in enumerate(rs):
            x = _rand_unit(d, rng)
            u = _orth_unit_to(x, rng)
            y = x + r * u
            # k-bit 量化（独立量化 x,y；与训练中的“逐张量缩放 + 随机舍入”一致）
            qx = _quant_vec_kbit_stoch(x, k, rng)
            qy = _quant_vec_kbit_stoch(y, k, rng)
            # 这里用同样的真值函数 g(r) 来评估“量化后的近似值”
            # 注意：论文没有公开具体近似式，这里采用与真值同一形式来衡量量化误差
            rq = np.linalg.norm(qx - qy)
            approx[i] = np.exp(- (rq / float(sigma))**2 )

        rmse = float(np.sqrt(np.mean((approx - true)**2)))

        # 作图（中文、加粗）
        fig, ax = plt.subplots(figsize=(6.2, 4.6))
        ax.scatter(rs, approx, s=6, alpha=0.45, label="量化近似")
        # 红线：按 r 分桶求真值均值（更平滑）
        bins = np.linspace(0.0, r_max, 40)
        idx = np.digitize(rs, bins)
        red_x, red_y = [], []
        for b in range(1, len(bins)+1):
            m = (idx == b)
            if np.any(m):
                red_x.append(rs[m].mean()); red_y.append(true[m].mean())
        ax.plot(np.array(red_x), np.array(red_y), 'r-', linewidth=2.2, label="真值（x,y）")

        ax.set_xlabel(r"$\|x-y\|_2$", fontweight='bold')
        ax.set_ylabel("相似度（归一化）", fontweight='bold')
        ax.set_title(f"FLQ，k={k}，RMSE={rmse:.4f}", fontweight='bold')
        leg = ax.legend(frameon=True); 
        for t in leg.get_texts(): t.set_fontweight('bold')
        ax.set_xlim(0, r_max); ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linewidth=0.6, alpha=0.4)
        ax.tick_params(labelsize=12, width=1.8)
        for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontweight('bold')
        fig.tight_layout()
        figs.append((fig, ax))
        print(f"[Fig4] k={k}  RMSE={rmse:.4f}")
    return figs


def main():
    import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt

    ap = argparse.ArgumentParser("FLQ 绘图：Fig.2 / Fig.3 / Fig.4")
    ap.add_argument("--excel_dir", type=str, default="results")
    ap.add_argument("--dataset", choices=["mnist","fmnist"], default="mnist")
    ap.add_argument("--modes", nargs="+", default=MODES_DEFAULT)  # 用于 Fig.2：["fedavg","laq8","bbit"]
    ap.add_argument("--prefix", type=str, default="results")
    ap.add_argument("--max_iter", type=int, default=800)
    ap.add_argument("--save", action="store_true")

    # Fig.2 样式与平滑
    ap.add_argument("--smooth_win", type=int, default=9)
    ap.add_argument("--smooth_ema", type=float, default=0.0)
    ap.add_argument("--legend_size", type=int, default=16)
    ap.add_argument("--line_width", type=float, default=2.5)

    # Fig.3 控制
    ap.add_argument("--do_fig3", action="store_true")
    ap.add_argument("--fig3_max_comm", type=int, default=200)

    # Fig.4 受控仿真参数（主角：调用 draw_paper_fig4_quant_controlled）
    ap.add_argument("--fig4_k", type=str, default="4,8")   # 逗号分隔，例如 "4,8"
    ap.add_argument("--fig4_d", type=int, default=64)
    ap.add_argument("--fig4_n", type=int, default=4000)
    ap.add_argument("--fig4_rmax", type=float, default=5.0)
    ap.add_argument("--fig4_sigma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # ---- 收集 Excel（供 Fig.2/3 使用）----
    excel_map = {}
    for m in args.modes:
        key = MODE_ALIASES.get(m.lower(), m.lower())
        path = os.path.join(args.excel_dir, f"{args.prefix}_{args.dataset}_{key}.xlsx")
        if os.path.exists(path):
            excel_map[key] = path

    # ---- Fig.2：迭代轮次 vs 交叉熵 ----
    if excel_map:
        # 确保输出目录存在
        os.makedirs("figures", exist_ok=True)
        
        fig2, _ = draw_paper_fig2_iter(
            excel_map, max_iter=args.max_iter,
            smooth_win=args.smooth_win, smooth_ema=args.smooth_ema,
            legend_size=args.legend_size, line_width=args.line_width
        )
        out2 = f"figures/Fig2_{args.dataset}.png"
        if args.save: fig2.savefig(out2, dpi=300); print(f"[save] {out2}")
        else: plt.show()

    # ---- Fig.3：通信中的二值量化结果（可选，需要 bin 的 Excel）----
    if args.do_fig3 and ("bin" in excel_map):
        fig3, _ = draw_paper_fig3_binary(excel_map["bin"], mode_key="bin", max_comm=args.fig3_max_comm)
        out3 = f"figures/Fig3_{args.dataset}.png"
        if args.save: fig3.savefig(out3, dpi=300); print(f"[save] {out3}")
        else: plt.show()

    # ---- Fig.4：直接在 main 里调用 draw_paper_fig4_quant_controlled ----
    # ks = [int(s) for s in args.fig4_k.split(",") if s.strip()]
    # figs = draw_paper_fig4_quant_controlled(
    #     ks=tuple(ks),
    #     d=args.fig4_d, n=args.fig4_n,
    #     r_max=args.fig4_rmax, sigma=args.fig4_sigma,
    #     seed=args.seed
    # )
    # # 分别保存 k=... 的子图
    # for k, (fig, _) in zip(ks, figs):
    #     out4 = f"figures/Fig4_k{k}_{args.dataset}.png"
    #     if args.save: fig.savefig(out4, dpi=300); print(f"[save] {out4}")
    #     else: plt.show()


if __name__ == "__main__":
    main()
