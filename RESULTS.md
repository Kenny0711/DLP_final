# DLP Final — 實驗結果分析 / Experiment Results Analysis

## 實驗設定 / Experiment Setup

- **環境**: SUMO 十字路口模擬（忠孝×復興南路口）
- **觀測空間**: 28-dim = `[queue/wait/speed × 8 edges] + [flow_E2C, flow_W2C, flow_N2C, flow_S2C]`
- **動作空間**: 2（保持 / 切換綠燈相位）
- **Context 切換**: morning（0–359 steps）→ off-peak（360–719）→ evening（720–1079）
- **訓練輪數**: 500 epochs（各算法）

### Context 車流設定 / Traffic Flows

| Context | EB vph | WB vph | NB vph | SB vph |
|---------|--------|--------|--------|--------|
| morning_peak | 319 | 489 | 138 | 153 |
| off_peak | 331 | 418 | 91 | 102 |
| evening_peak | 349 | **657** | 203 | 225 |

---

## 主要結果 / Main Results

### 最後 50 epochs 平均 reward（越高越好）

| 算法 | 整體 mean | 早峰 | 離峰 | **晚峰** | OOD 觸發率 |
|------|-----------|------|------|---------|-----------|
| Fixed-Time | -5.911 | -5.606 | -4.983 | -7.321 | — |
| A2C | -1.830 | -1.785 | -1.861 | -1.367 | — |
| LCPO（原始） | -1.830 | -1.785 | -1.861 | -1.367 | 0/500 = 0% |
| **LCPO（修正）** | **-1.741** | **-1.699** | **-1.760** | -1.231 | **446/500 = 89.2%** |
| DDQN | -1.835 | -1.786 | -1.858 | -1.251 | — |
| **GRU-LCPO** | -1.760 | -1.720 | -1.769 | **-1.050** | 431/500 = 86.2% |

**LCPO（修正後）整體 mean 最佳**（−1.741），比原始 LCPO/A2C 改善 5%。  
**GRU-LCPO 晚峰最佳**（−1.050），比 LCPO 修正版再改善 15%（−1.050 vs −1.231）。  
GRU 的潛在 Context 表示對高負載情境（晚峰 WB=657 vph）提供更強的適應能力。

> Reward = 負停止車輛數，越接近 0 越好。Fixed-Time 是固定 60s 週期，不做任何學習。

---

## OOD Detector 效果分析 / OOD Detector Analysis

| 算法 | OOD 觸發比例 | 觸發機制 | 說明 |
|------|------------|---------|------|
| LCPO（原始） | **0.0%** (0/500) | L2 on full obs (28-dim) | 閾值 `2√(var×28)` 受噪音維度主導，從未觸發 |
| **LCPO（修正）** | **89.2%** (446/500) | L2 on flow dims only (4-dim) | 僅用流量維度，信噪比高，有效偵測情境切換 |
| **GRU-LCPO** | **86.2%** (431/500) | L2 on h_t (32-dim) | GRU 在潛在空間自動放大情境差異 |

**修正效果**：僅把 OOD 公式從 28-dim 縮減到 4 個流量維度，LCPO 觸發率從 0% 躍升至 89.2%。

---

## 學習曲線趨勢 / Learning Curve Trends

### 早峰 reward（morning）

| 算法 | 前 50 epoch | 後 50 epoch | 改善 |
|------|------------|------------|------|
| LCPO | -2.325 | -1.785 | +0.540 |
| GRU-LCPO | -2.253 | -1.720 | **+0.533** |

### 晚峰 reward（evening）

| 算法 | 前 50 epoch | 後 50 epoch | 改善 |
|------|------------|------------|------|
| LCPO | -1.742 | -1.367 | +0.375 |
| GRU-LCPO | -1.875 | -1.050 | **+0.825** |

GRU-LCPO 在晚峰（最高車流量，最困難的 context）的改善幅度顯著更大。

---

## 圖表說明 / Figures

| 檔案 | 內容 |
|------|------|
| `figures/reward_comparison.png` | 五種算法整體 reward 比較（500 epochs 曲線） |
| `figures/context_comparison.png` | 各算法在三個 context 的 reward 分段比較 |
| `figures/context_reward_curves.png` | 各 context 的 reward 學習曲線（隨 epoch 變化） |
| `figures/adaptation_speed.png` | Context 切換後的適應速度分析 |
| `figures/forgetting.png` | Catastrophic forgetting 指標比較 |
| `figures/gru_latent_tsne.png` | GRU h_t 潛在 context 向量的 t-SNE 視覺化 |

---

## GRU Latent Space 分析 / GRU Latent Analysis

`figures/gru_latent_tsne.png`：對 3 個完整 episode（2580 步）的 h_t 向量做 t-SNE 降維。

加入 flow signal 後，GRU 學到的 h_t 具備 context 區分能力，OOD detector 能有效偵測 context 轉換。

---

## 結論 / Conclusions

1. **修正 LCPO OOD 後，整體 mean reward 最佳**（−1.741）：僅將 OOD 公式改為流量維度，觸發率從 0% 升至 89.2%，整體效能超越 GRU-LCPO。

2. **GRU-LCPO 晚峰效能最佳**（−1.050）：比修正版 LCPO 再改善 15%（vs −1.231），比 A2C 改善 23%（vs −1.367）。GRU 的潛在表示對高負載情境提供更強的情境適應。

3. **A2C = LCPO（原始）**：OOD 從未觸發時，KL 約束無效，LCPO 退化為普通 A2C。

4. **Flow signal 是根本解法**：在 obs 加入各方向進入車輛的滾動平均，使任何 OOD-based 算法都能區分三段情境。

5. **潛在 OOD 更通用**：LCPO 修正版需手動指定流量維度；GRU-LCPO 自動在 h_t 空間找到情境分離表示，不需要領域知識。

---

## 對應 LCPO 原論文 / Relation to LCPO Paper

| LCPO 論文設定 | 本實作 |
|-------------|--------|
| Context z_t = 風向量（直接觀測） | Context ≈ flow signal（進入車輛數滾動平均） |
| OOD on z_t（低維，易分） | OOD on h_t（GRU 潛在空間） |
| π(a｜s_t, z_t) | π(a｜[s_t, h_t]) |
| Wind gym experiment | SUMO 真實路口模擬 |

本研究將 LCPO 延伸至 **latent context** 設定（論文 §7 future work），在 h_t 空間做 OOD 偵測，驗證了其可行性。
