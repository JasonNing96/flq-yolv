#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLQ Federated Learning v4 - PyTorch Implementation
基于v3版本，将TensorFlow替换为PyTorch
"""
from __future__ import annotations
import argparse, os, numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# GPU配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"使用GPU: {torch.cuda.get_device_name()}")
    torch.backends.cudnn.benchmark = True
else:
    print("使用CPU")

# ------------------------- I/O -------------------------
def save_excel_data(outfile: str, mode: str, history: dict,
                    bin_series: np.ndarray | None = None,
                    grad_samples: np.ndarray | None = None):
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    iters = len(history["loss"])
    df = pd.DataFrame({
        "iter": np.arange(1, iters+1, dtype=np.int32),
        "loss": history["loss"].astype(np.float32),
        "acc":  history["acc"].astype(np.float32),
        "entropy": history["entropy"].astype(np.float32),
        "selcnt": history["selcnt"].astype(np.int32),
        "bits_up_cum": history["bits_up_cum"].astype(np.float64),
        "bits_down_cum": history["bits_down_cum"].astype(np.float64),
    })
    df["cum_bits_total"] = df["bits_up_cum"] + df["bits_down_cum"]
    with pd.ExcelWriter(outfile) as xw:
        df.to_excel(xw, sheet_name=f"curve_{mode}", index=False)
        if bin_series is not None:
            pd.DataFrame({"comm": np.arange(len(bin_series), dtype=np.int32),
                          "bit":  np.array(bin_series, dtype=np.int8)}
            ).to_excel(xw, sheet_name=f"bin_{mode}", index=False)
        if grad_samples is not None:
            pd.DataFrame({"gt": np.array(grad_samples, dtype=np.float32)}
            ).to_excel(xw, sheet_name=f"gt_{mode}", index=False)
    print(f"[save_excel_data] wrote {outfile}")

# ------------------------- tensor utils -------------------------
def weights_to_vec(model):
    """将模型参数转换为一维向量"""
    return torch.cat([p.data.view(-1) for p in model.parameters()]).cpu().numpy().astype(np.float32)

def vec_to_weights(vec, model):
    """将一维向量赋值给模型参数"""
    vec_tensor = torch.from_numpy(vec).to(device)
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec_tensor[idx:idx+n].view(p.shape))
        idx += n

def gradtovec(model):
    """将梯度转换为一维向量"""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.data.view(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=device))
    return torch.cat(grads).cpu().numpy().astype(np.float32)

def vectograd(vec, model):
    """将一维向量转换为梯度并应用"""
    vec_tensor = torch.from_numpy(vec).to(device)
    idx = 0
    for p in model.parameters():
        n = p.numel()
        if p.grad is None:
            p.grad = torch.zeros_like(p.data)
        p.grad.data.copy_(vec_tensor[idx:idx+n].view(p.shape))
        idx += n

# ------------------------- per-tensor quant helpers -------------------------
def _split_flat(vec: np.ndarray, shapes):
    out, off = [], 0
    for shp in shapes:
        n = int(np.prod(shp)); out.append(vec[off:off+n]); off += n
    return out

def _cat_flat(chunks):
    return np.concatenate(chunks, axis=0) if len(chunks) else np.zeros(0, np.float32)

def _quant_tensor_stoch(x: np.ndarray, b: int) -> np.ndarray:
    """对称k-bit量化，随机舍入"""
    L = float(2**(b-1) - 1)
    amax = float(np.max(np.abs(x))) + 1e-12
    s = amax / L
    y = x / s
    low = np.floor(y); p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q_int = np.clip(low + rnd, -L, L)
    return (q_int * s).astype(np.float32)

def _quant_bin_tensor(diff: np.ndarray) -> np.ndarray:
    """二值相对量化"""
    sgn = np.sign(diff)
    if np.any(sgn == 0):
        rnd = (np.random.rand(*sgn.shape) < 0.5).astype(np.float32)
        sgn = np.where(sgn == 0, 2.0 * rnd - 1.0, sgn)
    alpha = float(np.mean(np.abs(diff)))
    return alpha * sgn

def quant_rel_per_tensor(g_vec: np.ndarray, ref_vec: np.ndarray, b: int, shapes) -> np.ndarray:
    """逐张量相对域量化"""
    if b <= 0:
        return ref_vec.astype(np.float32)
    g_chunks = _split_flat(g_vec, shapes)
    ref_chunks = _split_flat(ref_vec, shapes)
    out = []
    if b == 1:
        for gt, rt in zip(g_chunks, ref_chunks):
            out.append(rt + _quant_bin_tensor(gt - rt))
    else:
        for gt, rt in zip(g_chunks, ref_chunks):
            out.append(rt + _quant_tensor_stoch(gt - rt, b))
    return _cat_flat(out)

def laq_per_vector(g_vec: np.ndarray, k: int) -> np.ndarray:
    """LAQ向量级量化"""
    L = float(2**(k-1) - 1)
    s = (np.max(np.abs(g_vec)) + 1e-12) / L
    y = g_vec / s
    low = np.floor(y); p = y - low
    rnd = (np.random.rand(*y.shape) < p).astype(np.float32)
    q = np.clip(low + rnd, -L, L) * s
    return q.astype(np.float32)

def laq_per_tensor(g_vec: np.ndarray, k: int, shapes) -> np.ndarray:
    """LAQ逐张量量化"""
    g_chunks = _split_flat(g_vec, shapes)
    out = [_quant_tensor_stoch(gt, k) for gt in g_chunks]
    return _cat_flat(out)

# ------------------------- datasets -------------------------
class FederatedDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def _load_arrays(name: str):
    """加载数据集"""
    name = name.lower()
    if name in ["mnist", "mn"]:
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif name in ["fmnist", "fashion", "fashion_mnist"]:
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"unknown dataset: {name}")
    
    # 转换为numpy数组
    train_data = train_dataset.data.float().unsqueeze(1) / 255.0  # [N, 1, 28, 28]
    train_targets = train_dataset.targets
    test_data = test_dataset.data.float().unsqueeze(1) / 255.0
    test_targets = test_dataset.targets
    
    return (train_data, train_targets), (test_data, test_targets)

def make_federated_iid(dataset: str, M: int, batch: int, seed: int = 1234):
    """IID数据分割"""
    (train_data, train_targets), (test_data, test_targets) = _load_arrays(dataset)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n = len(train_data)
    indices = torch.randperm(n)
    train_data = train_data[indices]
    train_targets = train_targets[indices]
    
    Mi = n // M
    dataloaders = []
    for m in range(M):
        start_idx = m * Mi
        end_idx = (m + 1) * Mi
        client_data = train_data[start_idx:end_idx]
        client_targets = train_targets[start_idx:end_idx]
        
        client_dataset = FederatedDataset(client_data, client_targets)
        dataloader = DataLoader(client_dataset, batch_size=batch, shuffle=True, drop_last=True)
        dataloaders.append(dataloader)
    
    test_dataset = FederatedDataset(test_data, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    return dataloaders, test_dataloader

def make_federated_non_iid(dataset: str, M: int, batch: int, alpha: float = 0.3, 
                          seed: int = 1234, max_tries: int = 100):
    """非IID数据分割（Dirichlet分布）"""
    (train_data, train_targets), (test_data, test_targets) = _load_arrays(dataset)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_classes = int(train_targets.max()) + 1
    rng = np.random.default_rng(seed)
    
    # 按类别分组
    per_class_idx = [torch.where(train_targets == c)[0].numpy().tolist() for c in range(num_classes)]
    
    def _sample_partition():
        for li in per_class_idx:
            rng.shuffle(li)
        P = rng.dirichlet([alpha] * M, size=num_classes)
        client_bins = [[] for _ in range(M)]
        for c in range(num_classes):
            idxs = per_class_idx[c]
            n_c = len(idxs)
            raw = P[c] * n_c
            cnt = np.floor(raw).astype(int)
            rem = n_c - int(cnt.sum())
            if rem > 0:
                order = np.argsort(raw - cnt)[::-1]
                for j in range(rem):
                    cnt[order[j % M]] += 1
            start = 0
            for m in range(M):
                k = int(cnt[m])
                if k > 0:
                    client_bins[m].extend(idxs[start:start+k])
                    start += k
        for m in range(M):
            rng.shuffle(client_bins[m])
        return client_bins
    
    bins = _sample_partition()
    tries = 1
    while tries < max_tries and min(len(b) for b in bins) < batch:
        bins = _sample_partition()
        tries += 1
    
    dataloaders = []
    for m in range(M):
        inds = np.array(bins[m], dtype=np.int64)
        if len(inds) < batch:
            extra = rng.integers(low=0, high=len(train_data), size=(batch-len(inds),), endpoint=False)
            inds = np.concatenate([inds, extra], axis=0)
        
        client_data = train_data[inds]
        client_targets = train_targets[inds]
        client_dataset = FederatedDataset(client_data, client_targets)
        dataloader = DataLoader(client_dataset, batch_size=batch, shuffle=True, drop_last=True)
        dataloaders.append(dataloader)
    
    test_dataset = FederatedDataset(test_data, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    return dataloaders, test_dataloader

# ------------------------- model -------------------------
class CNNModel(nn.Module):
    def __init__(self, l2: float = 5e-4):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.l2_reg = l2
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits
        return x
    
    def l2_loss(self):
        """计算L2正则化损失"""
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss

def build_model(l2: float, lr: float):
    model = CNNModel(l2=l2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

# ------------------------- training -------------------------
def run(args):
    # ----- 数据 -----
    j_star = None  # Fig.3观察的坐标
    part = getattr(args, "partition", "iid").lower()
    alpha = float(getattr(args, "dir_alpha", 0.3))
    
    if part == "iid":
        dataloaders, test_dataloader = make_federated_iid(args.dataset, args.M, args.batch, seed=args.seed)
    else:
        dataloaders, test_dataloader = make_federated_non_iid(args.dataset, args.M, args.batch, alpha=alpha, seed=args.seed)
    
    # 创建迭代器
    train_iters = []
    for dataloader in dataloaders:
        # 创建无限迭代器
        def infinite_dataloader(dl):
            while True:
                for batch in dl:
                    yield batch
        train_iters.append(infinite_dataloader(dataloader))
    
    # ----- 模型 -----
    model, optimizer = build_model(l2=args.cl, lr=args.lr)
    
    # 获取参数形状
    shapes = [tuple(p.shape) for p in model.parameters()]
    nv = sum(int(np.prod(s)) for s in shapes)
    
    # ----- 超参与模式 -----
    M = int(args.M); K = int(args.iters)
    mode = getattr(args, "mode", "bbit").lower()
    b_up = 1 if mode == "bin" else int(getattr(args, "b", 8))
    b_down = int(getattr(args, "b_down", 0))
    if b_down in (0, 32): b_down = 0
    
    # 懒惰相关
    D = int(getattr(args, "D", 10)); ck = float(getattr(args, "ck", 0.8))
    C = int(getattr(args, "C", 50)); warmup = int(getattr(args, "warmup", 50))
    thr_scale = float(getattr(args, "thr_scale", 1.0))
    clip_global = float(getattr(args, "clip_global", 0.0))
    
    # 预算选择
    sel_clients = int(getattr(args, "sel_clients", 0))
    up_budget_bits = float(getattr(args, "up_budget_bits", 0.0))
    
    # 按被选端放大步长
    scale_by_selected = bool(int(getattr(args, "scale_by_selected", 1)))
    sel_ref = float(getattr(args, "sel_ref", 1.0))
    
    # ----- 状态 -----
    ref_up = np.zeros((M, nv), np.float32)
    ref_down = np.zeros((M, nv), np.float32)
    ef_res = np.zeros((M, nv), np.float32)
    theta = np.zeros(nv, np.float32)
    dtheta_hist = np.zeros((nv, D), np.float32)
    
    # 历史能量权重
    ksi = np.zeros((D, D+1), np.float32)
    for d in range(D):
        ksi[d, 0] = 1.0
        for k in range(1, D+1):
            ksi[d, k] = 1.0 / float(d + 1)
    ksi *= ck
    
    e = np.zeros(M, np.float32); ehat = np.zeros(M, np.float32); clock = np.zeros(M, np.int32)
    
    # ----- 统计 -----
    loss_hist = np.zeros(K, np.float32)
    acc_hist = np.zeros(K, np.float32)
    entropy_hist = np.zeros(K, np.float32)
    selcnt_h = np.zeros(K, np.int32)
    bits_up_cum = np.zeros(K, np.float64); bits_down_cum = np.zeros(K, np.float64)
    cum_up = 0.0; cum_down = 0.0
    
    # Fig3用：每轮1个二值位
    bin_series = []
    
    print(f"开始训练: {args.dataset} {mode} M={M} K={K}")
    
    for k in range(K):
        # 历史能量me_k
        var = weights_to_vec(model)
        if k > 0:
            dtheta = var - theta
            dtheta_hist = np.roll(dtheta_hist, 1, axis=1); dtheta_hist[:, 0] = dtheta
        theta = var
        me_k = 0.0; col_limit = min(k, D); kk = min(k, D)
        for d in range(col_limit):
            col = dtheta_hist[:, d]; me_k += float(ksi[d, kk] * (col @ col))
        
        w_global = var.copy()
        loss_round = 0.0
        
        # 本轮候选缓存
        g_eff_buf = [None]*M; q_buf = [None]*M
        cand_idx, cand_gain, cand_cost = [], [], []
        
        # ---------- 本地计算与候选构建 ----------
        for m in range(M):
            # 广播（下发量化）
            if b_down > 0:
                w_down = quant_rel_per_tensor(w_global, ref_down[m], b_down, shapes)
                ref_down[m] = w_down; vec_to_weights(w_down, model)
            else:
                vec_to_weights(w_global, model)
            
            # 本地训练
            try:
                x, y = next(train_iters[m])
                x, y = x.to(device), y.to(device)
            except StopIteration:
                # 重新创建迭代器
                def infinite_dataloader(dl):
                    while True:
                        for batch in dl:
                            yield batch
                train_iters[m] = infinite_dataloader(dataloaders[m])
                x, y = next(train_iters[m])
                x, y = x.to(device), y.to(device)
            
            model.zero_grad()
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            l2_loss = model.l2_loss()
            loss = ce + l2_loss
            loss.backward()
            
            g = gradtovec(model)
            loss_round += float(loss.item()) / M
            
            if mode in ["bbit", "bin"]:
                g_eff = g + ef_res[m]
                q = quant_rel_per_tensor(g_eff, ref_up[m], b_up, shapes)
                g_eff_buf[m] = g_eff; q_buf[m] = q
                e[m] = float(np.dot(q - g_eff, q - g_eff))
                rhs = (me_k / (args.lr * args.lr * M * M)) + 3.0 * (e[m] + ehat[m])
                pass_thr = (float(np.dot(q, q)) >= thr_scale * rhs) or (k < warmup) or (clock[m] >= C)
                if pass_thr:
                    cand_idx.append(m)
                    cand_gain.append(float(np.dot(q, q)))
                    cand_cost.append(b_up * nv)
            elif mode == "laq8":
                q = laq_per_vector(g, 8)
                q_buf[m] = q
                cand_idx.append(m); cand_gain.append(float(np.dot(q, q))); cand_cost.append(8 * nv)
            else:  # fedavg
                q = g; q_buf[m] = q
                cand_idx.append(m); cand_gain.append(float(np.dot(q, q))); cand_cost.append(32 * nv)
        
        # 还原全局权重
        vec_to_weights(w_global, model)
        
        # ---------- 预算贪心选择 ----------
        order = np.argsort(np.array(cand_gain) / (np.array(cand_cost) + 1e-12))[::-1]
        selected = []
        if sel_clients > 0:
            selected = list(order[:min(sel_clients, len(order))])
        elif up_budget_bits > 0.0:
            budget = up_budget_bits
            for i in order:
                if cand_cost[i] <= budget:
                    selected.append(i); budget -= cand_cost[i]
        else:
            selected = list(order)
        
        # ---------- Fig3：记录本轮1个二值位（仅bin） ----------
        if mode == "bin":
            if len(selected) > 0:
                # 本轮聚合前的"候选和"，用于选j*
                agg_tmp = np.zeros(nv, np.float32)
                for idx in selected:
                    m = cand_idx[idx]; agg_tmp += q_buf[m]
                if j_star is None:
                    j_star = int(np.argmax(np.abs(agg_tmp)))  # 第一次确定后固定
                
                # 多数表决：看被选端在j*的相对更新符号
                votes = []
                for idx in selected:
                    m = cand_idx[idx]
                    diff_j = (q_buf[m][j_star] - ref_up[m][j_star])
                    votes.append(1.0 if diff_j >= 0.0 else -1.0)
                b = 1 if (np.mean(votes) >= 0.0) else 0
                bin_series.append(b)
            else:
                bin_series.append(bin_series[-1] if bin_series else 0)
        
        # ---------- 聚合与更新 ----------
        bits_up_this = 0.0
        if len(selected) > 0:
            agg = np.zeros(nv, np.float32)
            for idx in selected:
                m = cand_idx[idx]; q = q_buf[m]
                agg += q; bits_up_this += cand_cost[idx]
                if mode in ["bbit", "bin"]:
                    ref_up[m] = q
                    ef_res[m] = g_eff_buf[m] - q
                    ehat[m] = e[m]
                clock[m] = 0
            
            m_sel = float(len(selected))
            g_hat = agg / max(1.0, m_sel)
            scale_sel = (m_sel / max(1.0, sel_ref)) if scale_by_selected else 1.0
            
            # 应用梯度
            vectograd(g_hat * scale_sel, model)
            
            # 全局梯度裁剪
            if clip_global > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_global)
            
            optimizer.step()
        
        # 未选端
        picked = set(cand_idx[i] for i in selected)
        for m in range(M):
            if m in picked: continue
            if mode in ["bbit", "bin"] and g_eff_buf[m] is not None:
                ef_res[m] = g_eff_buf[m]
            clock[m] = min(clock[m] + 1, C + 1)
        
        # ----- 比特统计 -----
        selcnt = len(selected)
        bits_down_this = (b_down if b_down > 0 else 32) * nv * selcnt
        cum_up += bits_up_this; cum_down += bits_down_this
        
        # ----- 测试 -----
        if (k + 1) % 10 == 0 or k == K - 1:
            model.eval()
            correct = 0; total = 0; ent_sum = 0.0
            with torch.no_grad():
                for x, y in test_dataloader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    pred = logits.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
                    ent_sum += F.cross_entropy(logits, y, reduction='sum').item()
            
            acc = correct / total
            entropy = ent_sum / total
            model.train()
        else:
            acc = acc_hist[k-1] if k > 0 else 0.0
            entropy = entropy_hist[k-1] if k > 0 else 0.0
        
        # ----- 记录与打印 -----
        loss_hist[k] = loss_round
        acc_hist[k] = acc
        entropy_hist[k] = entropy
        selcnt_h[k] = selcnt
        bits_up_cum[k] = cum_up
        bits_down_cum[k] = cum_down
        
        if (k + 1) % 10 == 0 or k < 5 or k == K - 1:
            print(f"[{k+1}/{K}] acc={acc_hist[k]:.4f} entropy={entropy_hist[k]:.4f} "
                  f"sel={selcnt}/{len(cand_idx)}/{M} | bits↑Σ={cum_up:.2e} bits↓Σ={cum_down:.2e}")
    
    return {
        "loss": loss_hist, "acc": acc_hist, "entropy": entropy_hist,
        "selcnt": selcnt_h, "bits_up_cum": bits_up_cum, "bits_down_cum": bits_down_cum,
        "bin_series": np.array(bin_series, dtype=np.int8) if mode == "bin" else None
    }

# ------------------------- main -------------------------
def parse_args():
    p = argparse.ArgumentParser("FLQ v4 PyTorch implementation")
    # data
    p.add_argument("--dataset", type=str, default="mnist",
                   choices=["mnist", "fmnist", "fashion", "fashion_mnist"])
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--partition", type=str, default="non_iid", choices=["iid","non_iid"])
    p.add_argument("--dir_alpha", type=float, default=0.1)
    # modes
    p.add_argument("--mode", type=str, default="bbit", choices=["fedavg","bbit","bin","laq8"])
    p.add_argument("--b", type=int, default=8, help="uplink bits; bin forces 1")
    p.add_argument("--b_down", type=int, default=8, help="downlink bits; 0/32 = FP32")
    # opt
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cl", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_global", type=float, default=0.0)
    # budget
    p.add_argument("--sel_clients", type=int, default=0)
    p.add_argument("--up_budget_bits", type=float, default=17000000.0)
    # lazy
    p.add_argument("--D", type=int, default=10)
    p.add_argument("--ck", type=float, default=0.8)
    p.add_argument("--C", type=int, default=1000000000)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--thr_scale", type=float, default=0.0)
    p.add_argument("--scale_by_selected", type=int, default=1)
    p.add_argument("--sel_ref", type=float, default=1.0)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    hist = run(args)
    outfile = f"results/results_{args.dataset}_{args.mode}_{args.iters}.xlsx"
    save_excel_data(outfile, args.mode, hist, bin_series=hist.get("bin_series"))
    print(f"训练完成！结果已保存到: {outfile}")
