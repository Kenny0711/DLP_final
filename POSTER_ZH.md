# 基於流量率 Context 訊號的 GRU-LCPO 自適應交通號誌控制

**課程**：深度學習實作（DLP）— 期末專題，國立陽明交通大學，2026 春季

---

## 摘要

交通號誌控制必須適應多種交通情境（早峰、離峰、晚峰）。**LCPO（ICLR 2025）**透過對 $z_t$（如風向量）的分布外偵測（OOD）施加 KL 約束，防止策略在情境切換時災難性遺忘。然而交通環境缺乏顯式 $z_t$：當策略控制良好時，排隊/等待/速度在三種情境下高度重疊，OOD 偵測失效。

本研究提出兩項貢獻：**(1) 流量率 Context 訊號**（60-step 滾動平均進入車輛數，obs dim 24→28），作為可觀測的情境代理；**(2) GRU 編碼器**，從觀測歷史學出 $h_t \approx z_t$，在潛在空間執行 OOD 偵測，無需情境標籤。

同時識別並修正 LCPO 的**閾值縮放問題**（$d=28$ 導致閾值過高，OOD 觸發率 0%）。LCPO（修正）達到最佳整體平均 reward（−1.741）；GRU-LCPO 達到最佳晚峰 reward（−1.050），**比 A2C 改善 23%**，晚峰學習效率是 A2C 的 **2.2 倍**。

---

## 1. 動機

### 為什麼現有觀測值無法作為 Context 訊號

![Motivation](figures/poster_fig0_motivation.png)

當策略控制良好時，排隊長度與等待時間在三種情境下分佈高度重疊，無法區分。唯有**車輛到達率（流量）**能可靠地分離情境（離峰→晚峰 +57%）。

| 訊號 | 早峰 | 離峰 | 晚峰 | 可區分？ |
|------|------|------|------|---------|
| 西向排隊長度 | ~2.1 輛 | ~1.8 輛 | ~2.3 輛 | **否**（重疊） |
| 西向等待時間 | ~3.2 s | ~2.9 s | ~3.5 s | **否**（重疊） |
| **西向車輛數（流量）** | **489 vph** | **418 vph** | **657 vph** | **是（+57%）** |

每個 episode = 1080 steps，分為 3 段（早峰 / 離峰 / 晚峰）。我們在 24-dim 狀態上附加 4-dim 流量訊號（60-step 滾動均值，dims 24–27），將觀測擴充為 28-dim。

---

## 2. 方法

### 2.1 GRU 學習隱式 Context $z_t$

LCPO 原論文中 $z_t$ 可直接觀測（如風向量）。本研究**從觀測歷史學習** $h_t \approx z_t$：

$$h_t = \text{GRU}(s_{t-k}, \ldots, s_t) \in \mathbb{R}^{32}$$

$$\pi(a \mid [s_t,\; h_t]), \quad V([s_t,\; h_t]) \quad \text{（60 維增廣觀測）}$$

在 $h_t$ 潛在空間執行 OOD 偵測：

$$\text{OOD}(h_t) = \|h_t - \bar{h}_{\text{recent}}\|_2 > 2\sqrt{\hat{\sigma}^2_{h} \cdot 32}$$

### 2.2 LCPO OOD 閾值問題（已修正）

原始公式在 $d=28$ 時：

$$\text{閾值} = 2\sqrt{\text{var}_{\text{mean}} \times 28}$$

24 個噪音維度（排隊/速度）透過 $\sqrt{28}\approx 5.3\times$ 膨脹閾值 → **觸發率 0%**。

**修正**：僅使用流量維度（$d_{\text{flow}}=4$，情境內方差低、情境間方差高）：

$$\text{閾值}_{\text{修正}} = 2\sqrt{\text{var}_{\text{flow}} \times 4} \quad \Rightarrow \quad \text{觸發率 89.2\%}$$

### 2.3 訓練流程

- **BC 熱身**（50 epochs）：GRU 從 A2C 軌跡序列學習動作預測，無需情境標籤
- **線上 GRU-LCPO**（500 epochs）：在 $h_t$ 空間執行 OOD 偵測的 LCPO 約束更新

---

## 3. 實驗結果

### 3.1 GRU 學到了什麼？—— 學習動態分析

![Evening Peak Learning Curve](figures/poster_fig8_evening_learning.png)

A2C 與 GRU-LCPO 從相近初始水準（約 −1.87）出發。500 epochs 後，GRU-LCPO 晚峰改善 **+0.825**，A2C 僅改善 **+0.374**——最難情境的學習效率是 A2C 的 **2.2 倍**。

差距源於 $h_t$ 在情境切換時觸發 LCPO 的 KL 約束，防止策略切換回早峰時遺忘晚峰的應對方式。

### 3.2 整體收斂曲線

![Overall Learning Curves](figures/poster_fig1_learning_curves.png)

LCPO（修正）收斂最快，整體平均 reward 最佳。GRU-LCPO 前期因 BC 熱身（0–50 epochs）起點較低，約 100 epochs 後超越 A2C 與 DDQN。兩種方法互補：LCPO（修正）整體 mean 最佳；GRU-LCPO 在最難情境（晚峰）最佳（見 §3.1）。

### 3.3 OOD 偵測器分析

![OOD Trigger Rate](figures/poster_fig3_ood_rate.png)

所有 baseline 共用相同的 28-dim 輸入（包含流量訊號）；LCPO/GRU-LCPO 的優勢純粹來自情境觸發的 KL 約束，而非資訊優勢。沒有 OOD 偵測器的 A2C 對所有 step 一視同仁更新，在早峰/離峰大量更新時逐漸覆蓋晚峰應對策略。

將 LCPO 套用於完整 28-dim 觀測（無顯式 $z_t$）會使觸發率降為 **0%**——等同於 A2C——因為 24 個噪音狀態維度透過 $\sqrt{28} \approx 5.3\times$ 膨脹閾值。

| 偵測器 | OOD 空間 | 觸發率 |
|--------|---------|--------|
| **LCPO（修正，本研究）** | 流量維度作為 $z_t$ 代理（4 維） | **89.2%** |
| **GRU-LCPO（本研究）** | 學習到的 $h_t$ 作為 $z_t$（32 維潛在） | **86.2%** |

### 3.4 主要結果

![Per-Context Bar Chart](figures/poster_fig2_context_bar.png)

| 算法 | 平均 | 早峰 | 離峰 | **晚峰** | OOD 觸發率 |
|------|------|------|------|---------|-----------|
| Fixed-Time | −5.911 | −5.606 | −4.983 | −7.321 | — |
| A2C | −1.830 | −1.785 | −1.861 | −1.367 | — |
| **LCPO（修正）** | **−1.741** | **−1.699** | **−1.760** | −1.231 | 89.2% |
| DDQN | −1.835 | −1.786 | −1.858 | −1.251 | — |
| **GRU-LCPO** | −1.760 | −1.720 | −1.769 | **−1.050** | 86.2% |

> Reward = 每步停止車輛數的負值，越高越好。最後 50 epoch 平均。

**LCPO（修正）**：整體 mean 最佳——僅修正 OOD 閾值即從 −1.830（= A2C）改善至 −1.741。  
**GRU-LCPO**：晚峰最佳（−1.050），比 LCPO（修正）**再好 15%**，比 A2C 好 **23%**。

---

## 4. 結論

1. **排隊/等待/速度無法作為 LCPO 的 $z_t$**：當策略良好時三種情境高度重疊。流量率訊號（57% 變幅）提供可靠的情境區分能力。

2. **GRU 學到隱式 $z_t$**：$h_t$ 從觀測歷史替代原論文的手工 $z_t$。證據：86.2% OOD 觸發率；晚峰改善幅度是 A2C 的 **2.2 倍**（+0.825 vs +0.374）。

3. **LCPO 閾值縮放問題**：$2\sqrt{\text{var}\times d}$ 在 $d$ 大且有噪音維度時失效。改用 4 個流量維度可將觸發率從 0% 提升至 89.2%，整體 reward 改善 5%。

4. **潛在 OOD 更通用**：流量 OOD 需領域知識（手動選維度）；GRU-LCPO 自動學習——可推廣至任何情境嵌於觀測動態的環境。

---

## 5. 參考文獻

1. Luo, D. et al. (2025). *Locally Constrained Policy Optimization for Online Reinforcement Learning*. ICLR 2025.
2. Krajzewicz, D. et al. (2012). *Recent Development and Applications of SUMO*. IJATS 5(3&4).
3. Mnih, V. et al. (2016). *Asynchronous Methods for Deep Reinforcement Learning*. ICML 2016.
4. Van Hasselt, H. et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI 2016.
5. Schulman, J. et al. (2015). *Trust Region Policy Optimization*. ICML 2015.

---

## 附錄 A：超參數設定

| 參數 | A2C | LCPO | DDQN | GRU-LCPO |
|------|-----|------|------|----------|
| 觀測維度 | 28 | 28 | 28 | 60 (28+32) |
| 隱藏層 | [128,128] | [128,128] | [128,128] | [128,128] |
| 學習率 | 3e-4 | 3e-4 | 1e-4 | 3e-4 |
| γ / λ | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / — | 0.99 / 0.95 |
| KL_in / KL_out | — | 0.01 / 0.05 | — | 0.01 / 0.05 |
| GRU 隱藏 / Context 維度 | — | — | — | 64 / 32 |
| 歷史窗口 k | — | — | — | 10 |
| BC 熱身 epochs | — | — | — | 50 |
| 訓練 epochs | 500 | 500 | 500 | 500 |

## 附錄 B：其他圖表

| 檔案 | 說明 |
|------|------|
| `poster_fig5_episode_structure.png` | Episode 時間軸與 3 段情境及 vph 標示 |
| `gru_latent_tsne.png` | GRU $h_t$ PCA & t-SNE（平衡採樣，3 種情境） |
| `poster_fig_ht_drift.png` | Episode 內 $h_t$ OOD 距離變化（860 steps） |
| `poster_fig4_evening_peak.png` | 晚峰效能水平排名 |
| `poster_fig6_cost_breakdown.png` | 各情境停止成本疊加分解 |
| `poster_fig10_learning_gain.png` | 各算法各情境學習進步量 |
| `poster_fig9_ood_timeline.png` | OOD 觸發率隨訓練過程變化 |
