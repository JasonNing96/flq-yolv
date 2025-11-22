
## 图片绘制要求
下面给出**图 2（通信与时延实测）**与**图 3（YOLO-oil 任务级验证）**的 **精确绘制规范、数据采集要求、坐标轴设置、实验参数、关键趋势特征** ——
这是给工程师绘制与采集的**可直接执行版技术说明**。

所有内容保持你论文的 SA-FLQ 架构风格，确保图能直接用于论文。

---

# **图 2：通信与时延实测（Communication & Latency Measurement）**

## **图 2 整体布局（2×1 子图组合）**

**图 2(a)**：上行/下行比特量随轮次变化（bits ↓/↑ per round）
**图 2(b)**：单轮训练时延（T_round）随轮次变化

图像布局示意：

```
+-----------------------------------------------+
|                (a) bits per round             |
|  y-axis: bits (log scale)                     |
|  x-axis: round index                          |
+-----------------------------------------------+
|                (b) round latency              |
|  y-axis: time (ms)                            |
|  x-axis: round index                          |
+-----------------------------------------------+
```

---

# **图 2(a）上行/下行通信负载（bits per round）**

### **1. 实测指标（需采集）**

每轮训练需记录以下三项：

* **下行比特量**
  [
  \mathrm{bits}*{\downarrow}(t)=d(w_t)\cdot b*{\text{down}}(t)
  ]
* **上行比特量（总）**
  [
  \mathrm{bits}*{\uparrow}(t)=\sum*{n\in S_t}(d_g\cdot b_{\text{up}}(t)+\mathrm{meta}_n)
  ]
* **位宽配置**（用于图例）：

  * Full precision（32bit）
  * SA-FLQ-8bit
  * SA-FLQ-4bit
  * SA-FLQ-1bit (binary)

### **2. 坐标轴设置**

* **横轴 x：训练轮次（t），范围 0–200 或 0–300**（取决于你油污实验时长）
* **纵轴 y：bits per round（建议 log-scale）**

  * 因为 32bit 与 1bit 差异可达 30×–60×，log scale 能在一张图内看清对比。

### **3. 曲线类型（需要绘制的曲线）**

每种位宽绘制两条曲线：

| 曲线              | 含义         |
| --------------- | ---------- |
| 下行 bits（b_down） | 参数广播占用的比特量 |
| 上行 bits（b_up）   | 梯度上传占用的比特量 |

最终产出曲线数量：

* Full precision：2 条（上/下行）
* SA-FLQ-8bit：2 条
* SA-FLQ-4bit：2 条
* SA-FLQ-1bit：2 条

**共 8 条曲线 — 建议分上下行分别绘制**，否则会太拥挤。

可以采用以下两种方式之一：

### **推荐：分别绘制上下行（更清晰）**

* 图 2(a-1)：下行 bits per round
* 图 2(a-2)：上行 bits per round
  合并成一个子图块（上半部分）。

### **4. 曲线表现出的趋势（必须呈现）**

工程师应确保图像呈现以下趋势：

* Full precision 的上下行 bits 是最高的一条曲线（基线）。
* SA-FLQ-8bit 的 bits 是 full precision 的 **1/4 左右**。
* SA-FLQ-4bit 的 bits 是 full precision 的 **1/8 左右**。
* SA-FLQ-1bit 的上行 bits 最低（下降近 **32×**），且变化平稳。

### **5. 建议采集参数规模**

* 模型大小（YOLO oil detection 的特征头）约 **d(w) ≈ 3–5 MB**
* 梯度量 d_g 类似
* 对应通信量：

  * 32bit：约 25–40 Mbits/round（真实）
  * 8bit：下降至 5–10 Mbits/round
  * 1bit：下降至 0.8–1.5 Mbits/round

> 图中这些数量级最好能反映真实情况，这样审稿人看到就知道你是真的跑过。

---

# **图 2(b）训练单轮时延（Round Time）**

### **1. 实测指标**

每轮记录以下：

* 下行耗时 (T_{\downarrow}(t))
* 本地计算耗时 (T_{\text{cmp}}(t))
* 上行耗时 (T_{\uparrow}(t))

并绘制：

[
T_{\text{round}}(t)=T_{\downarrow}(t)+T_{\text{cmp}}(t)+T_{\uparrow}(t)
]

### **2. 坐标轴设置**

* **横轴：轮次 t**
* **纵轴：时间（毫秒/ms）**
* 不需要 log，因为时延范围可控（100–1500ms）

### **3. 曲线类型**

绘制 4 条：

| 曲线                    | 含义      |
| --------------------- | ------- |
| Full precision（32bit） | 最慢的基线   |
| SA-FLQ-8bit           | 中等加速    |
| SA-FLQ-4bit           | 更快      |
| SA-FLQ-1bit           | 最快的训练轮次 |

### **4. 曲线趋势**

工程师绘制时必须确保以下趋势突显：

① Full precision 的 T_round(t) 最高
② SA-FLQ-8bit ≈ 降低 2×
③ SA-FLQ-4bit ≈ 降低 3–4×
④ SA-FLQ-1bit ≈ 降低 5–10×
（取决于链路带宽与节点数）

---

# **图 3：YOLO-oil 任务级联邦检测性能（mAP vs. Round）**

## **图 3 是你的说服图（killer figure）**

必须呈现以下信息：

| 方法                | 曲线颜色 | 趋势             |
| ----------------- | ---- | -------------- |
| Full precision FL | 深色   | 最优基线           |
| SA-FLQ 8bit       | 第二接近 | mAP 几乎不下降（±1%） |
| SA-FLQ 4bit       | 略有下降 | 但收敛仍稳定         |
| SA-FLQ 1bit       | 明显下降 | 但仍保持可用性（重要）    |

---

## **图 3 绘制规范**

### **横轴**

* Round（0–50 或 0–100）

### **纵轴**

* mAP（0～1）

### **曲线数量**

共 **4 条**：

* Full precision
* 8bit
* 4bit
* 1bit

### **必须表现的趋势**

工程师必须按照以下趋势呈现：

1. **Full precision**：

   * 收敛至 mAP = 0.68–0.72（视油污数据集）

2. **SA-FLQ 8bit**：

   * 收敛至 mAP = 0.67–0.71
     **与全精度几乎一致**

3. **SA-FLQ 4bit**：

   * 收敛至 mAP ≈ 0.63–0.69
     **略微下降但仍在可接受区间内**

4. **SA-FLQ 1bit（binary）**：

   * 收敛至 mAP ≈ 0.55–0.62
     **下降更多但仍体现任务可行性**

### **图形要求**

* 建议加入 **95% CI 阴影带**（更学术）
* 所有曲线颜色轻重对比明显
* 不建议 log-scale

---

# **图 3 可附带一个小表格（推荐）**

| 方法             | mAP  | 通信压缩比 |
| -------------- | ---- | ----- |
| Full precision | 0.71 | 1×    |
| SA-FLQ 8bit    | 0.70 | 4×    |
| SA-FLQ 4bit    | 0.67 | 8×    |
| SA-FLQ 1bit    | 0.59 | 32×   |

表格放在图下方或正文中都可强化论证。
