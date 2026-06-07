# DLP Final — Code Changes & Experiment Design Notes

## 本次修改摘要 / Summary of Changes

### 核心問題 / Root Problem

原版實作（obs 24 維）的 OOD detector 完全失效，因為：

1. Policy 觀測值是 `[queue_len, waiting_time, speed]` — 這是控制**結果**，不是 context 本身
2. 當 policy 控制良好時，三個 context（早峰/離峰/晚峰）的觀測分布高度重疊
3. GRU 學到的 h_t 向量在 t-SNE 上無法分群（各 context 之間距離 0.016–0.067，遠小於 OOD 閾值 0.21）

**根本原因**：LCPO 原論文把 **context 直接加進觀測**（如風向量），但我們的環境只有 queue/wait/speed，context（到達率）沒有暴露給 agent。

---

### 解決方案 / Solution

**在觀測中加入流量訊號（Flow-rate Context Signal）**

把各進入方向的滾動平均車輛數加入 obs，作為近似 vph（每小時車輛數）的 context 訊號。

這和 LCPO 論文把風向量加入觀測**完全類比**：
- LCPO 論文：obs = `[s_t, wind_vector]`（直接觀測 context）
- 本實作：obs = `[queue/wait/speed × 8, flow_E2C, flow_W2C, flow_N2C, flow_S2C]`

現實基礎：真實路口有感應式線圈（induction loop），可以測量通過車輛數。

| Context | WB（E2C）vph | OD 能偵測？|
|---------|------------|-----------|
| morning_peak | 489 | ✓ |
| off_peak | **418**（降 14%） | ✓ |
| evening_peak | **657**（升 57%） | ✓ |
| sudden_burst | **978**（升 2×） | ✓ |

---

## 修改的檔案 / Modified Files

### 1. `LCPO/sumo_intersection/sumo_env.py`

**obs_dim: 24 → 28**

新增 4 維流量訊號：

```python
_FLOW_EDGES = ["E2C", "W2C", "N2C", "S2C"]  # 4 incoming edges
_FLOW_WIN   = 60    # rolling window (60 steps ≈ 10 minutes)
_FLOW_MAX   = 20.0  # normalization constant
```

在 `_get_obs()` 中，維護 60-step 滾動平均的進入車輛數，正規化後附加到觀測：

```
obs[0:24]  = [queue/wait/speed × 8 edges]  (原有)
obs[24:28] = [flow_E2C, flow_W2C, flow_N2C, flow_S2C] / 20.0  (新增)
```

### 2. 訓練腳本（全部）

所有腳本的 `OBS_DIM: 24 → 28`：

| 檔案 | 修改 |
|------|------|
| `scripts/collect_warmup.py` | OBS_DIM 24→28 |
| `scripts/train_a2c.py` | OBS_DIM 24→28；改用 `sumo_env.SumoIntersectionEnv`（統一環境） |
| `scripts/train_lcpo.py` | OBS_DIM 24→28 |
| `scripts/train_ddqn.py` | OBS_DIM 24→28 |
| `scripts/train_gru_lcpo.py` | OBS_DIM 24→28；AUG_DIM 56→60 |
| `scripts/plot_gru_latent.py` | OBS_DIM 24→28；AUG_DIM 56→60 |

### 3. `scripts/train_a2c.py` 環境統一

原本 `train_a2c.py` 使用 `sumo_env_revised.py`（TDX API 真實資料），其他腳本使用 `sumo_env.py`（模擬資料）。

**改為全部使用 `sumo_env.py`**，原因：

收集的真實 TDX 資料（`aggregate_traffic.json`）三個時段的數值**完全相同**（只採集了 2 天，時段估算有誤），無法區分 context，不適合使用。

---

## 環境版本說明 / Environment Versions

| 檔案 | 路網 | 車流 | 觀測 |
|------|------|------|------|
| `sumo_env.py`（採用） | `intersection.net.xml` | `intersection.rou.xml`（模擬，三段 context） | 28-dim |
| `sumo_env_revised.py`（棄用） | `cross.net.xml` | TDX API 動態產生 | 24-dim（三段數值相同） |

---

## 重新訓練執行順序 / Training Execution Order

```bash
cd /home/mizu/2026spring_DLP_final/DLP_final

# Step 0: 重新收集 warmup（obs 維度已改，舊 checkpoint 不相容）
python3 scripts/collect_warmup.py

# Step 1–4: 四個算法（可分給四台機器平行跑）
python3 scripts/train_a2c.py
python3 scripts/train_lcpo.py
python3 scripts/train_ddqn.py
python3 scripts/train_gru_lcpo.py

# Step 5: 分析圖表
python3 scripts/plot_comparison.py
python3 scripts/plot_adaptation.py
python3 scripts/plot_gru_latent.py
```

---

## 預期改善 / Expected Improvements

加入流量訊號後：

1. **OOD detector（LCPO 和 GRU-LCPO）**：偵測到 context 切換時（step 360, 720），flow signal 會在 ~60 步內明顯改變（rolling window 緩衝），觸發 OOD buffer 更新。

2. **GRU 的 h_t 分群**：GRU encoder 現在有足夠的 signal（flow 隨時段明顯變化）可以學到不同 context 的表示，t-SNE 應該能看到明顯分群。

3. **公平比較**：所有算法（A2C、LCPO、DDQN、GRU-LCPO）使用相同的 28-dim obs，排除資訊不對等。

4. **A2C/DDQN 也受益**：雖然沒有 OOD，但 flow signal 讓 policy 可以學到「高流量時段 → 更積極清除 WB 方向」。

---

## 論文論述框架 / Paper Framing

本實作對應 LCPO 論文中對 context 的處理方式：

> *"We append the context vector z_t to the state s_t to form the augmented state [s_t, z_t]"* (LCPO §3)

我們的 flow signal 就是 z_t 的近似：從真實 SUMO 模擬中讀取的進入車輛數，扮演「可觀測的外生 context signal」角色。

GRU-LCPO 的創新是：
- 從 flow signal（弱 context signal）+ 觀測序列，用 GRU 學到更強的潛在表示 h_t
- 不需要 context 標籤，OOD 在 h_t 空間偵測

這對應 LCPO 論文 §7 提到的 future work：*"Combining LCPO with latent context inference"*
