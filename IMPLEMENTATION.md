# DLP Final — Implementation Notes / 實作說明

## 執行順序 / Execution Order

```
# 訓練 Training
Step 0 (所有人先跑)   python scripts/collect_warmup.py
Step 1 (Person 1)    python scripts/train_a2c.py
Step 2 (Person 2)    python scripts/train_lcpo.py
Step 3 (Person 3)    python scripts/train_ddqn.py
Step 4 (Person 4)    python scripts/train_gru_lcpo.py
Step 5 (Baseline)    python scripts/train_fixed_time.py

# 分析 Analysis (所有結果到齊後)
python scripts/plot_comparison.py       # overall reward comparison
python scripts/plot_adaptation.py       # adaptation speed after context switch
python scripts/plot_gru_latent.py       # GRU h_t t-SNE visualization

# 突發事件測試 Sudden burst test
python scripts/train_fixed_time.py --sudden
```

---

## 環境與場景設定 / Environment Setup

| 檔案 | 用途 | SUMO 時長 | Env steps |
|---|---|---|---|
| `nets/warmup.rou.xml` | Warm-up 場景（僅早峰） | 3600s | 360 |
| `nets/intersection.rou.xml` | 正式實驗（三段 context） | 10800s | 1080 |
| `nets/sudden.rou.xml` | 突發事件場景（四段，EB/WB×2） | 12600s | 1260 |

**車流資料來源 / Traffic data sources:**
- 忠孝x復興南：`traffic_cache.json`（14 天，兩個路口合併中位數）
- 忠孝x敦化：`traffic_cache_dunhua.json`

**Context 切換時間點 / Context transition points:**

| Context | SUMO time | Env steps | vph (EB/WB/NB/SB) |
|---|---|---|---|
| morning_peak | 0–3600s | 0–359 | 319 / 489 / 138 / 153 |
| off_peak | 3600–7200s | 360–719 | 331 / 418 / 91 / 102 |
| evening_peak | 7200–10800s | 720–1079 | 349 / 657 / 203 / 225 |
| **sudden_burst** | **10800–12600s** | **1080–1259** | **698 / 978 / 305 / 338** (EB/WB ×2) |

---

## 訓練流程說明 / Training Pipeline

### Step 0 — collect_warmup.py（共用）

用 A2C 在 **早峰場景**（warmup.sumocfg, 固定流量, 360 steps/episode）訓練 200 epochs。
同時收集 trajectory 存成 `results/warmup/trajectory.pkl`。

Train A2C on morning-peak only (simplest, single context).
Saves model checkpoint + full trajectory for downstream algorithms.

```
Warm-up A2C ──→ results/warmup/checkpoint.pt    (A2C weights)
             ──→ results/warmup/trajectory.pkl   (72,000 transitions)
```

---

### Step 1 — train_a2c.py（Person 1）

**A2C (baseline)**：直接繼承 warm-up 權重，在三段 context 場景繼續訓練。  
This is the **baseline without any forgetting mitigation**. Expected to show performance drop when context shifts.

```
checkpoint.pt → A2C → results/a2c/log.csv
```

---

### Step 2 — train_lcpo.py（Person 2）

**LCPO**：繼承 A2C 權重（完全相容），改用 TRPO 約束更新。
OOD buffer 偵測原始觀測空間中的 context shift。

Direct weight transfer from A2C (same FCNPolicy + FullyConnectNN architecture).
LCPO adds KL constraints to prevent the policy from drifting on past-context states.

```
checkpoint.pt → LCPO (KL_IN=0.01, KL_OUT=0.05) → results/lcpo/log.csv
```

---

### Step 3 — train_ddqn.py（Person 3）

**DDQN**：A2C 權重**不能直接用**（Q-network 架構不同）。  
改用 A2C trajectory 填 replay buffer 做**離線預訓練**，再做線上 ε-greedy 訓練。

Cannot reuse A2C weights (different network architecture).
Uses A2C trajectory for offline pre-training (fills replay buffer, 5000 SGD steps).
Then switches to online ε-greedy exploration.

```
trajectory.pkl → Replay Buffer (pre-fill)
              → Offline DQN SGD (5000 steps)
              → Online ε-greedy → results/ddqn/log.csv
```

---

### Step 4 — train_gru_lcpo.py（Person 4）

**GRU-LCPO（提出方法）**：兩階段。

Phase 1 — Behaviour Cloning warm-up（BC 熱身）:
- 把 trajectory 重建成長度 k=10 的觀測序列
- 用 cross-entropy loss 訓練 GRU encoder + policy 模仿 A2C 的動作
- A2C 權重**不能直接載入**（輸入維度不同：24 vs 56）
- BC 訓練 50 epochs，讓 GRU 學會從序列提取 context

Phase 2 — Online GRU-LCPO:
- OOD 偵測在 **h_t 潛在空間**（非原始觀測空間）
- 不需要手動 context 標籤

```
trajectory.pkl → BC warm-up (50 epochs, GRU encoder + policy)
              → Online GRU-LCPO → results/gru_lcpo/log.csv
```

---

## GRU-LCPO 架構說明 / GRU-LCPO Architecture

```
x_{t-k}, ..., x_t  →  GRU(24→64)  →  hidden  →  Linear(64→32)  →  tanh  →  h_t [32]
                                                                              ↓
                              s_t [24]  →  concat([s_t, h_t])  →  aug_obs [56]
                                                                      ↓
                                                              Policy πθ  →  action
                                                              Value  Vφ  →  V(s)
```

**Key differences from vanilla LCPO / 與原版 LCPO 的差異：**

| Component | Vanilla LCPO | GRU-LCPO |
|---|---|---|
| Policy input | s_t (24-dim) | [s_t, h_t] (56-dim) |
| Value input | s_t (24-dim) | [s_t, h_t] (56-dim) |
| OOD detection | raw obs space | h_t latent space |
| Context labels | Manual required | Not needed |
| Extra module | — | GRUEncoder(24→64→32) |

---

## 結果彙整 / Result Aggregation

各台機器把 `results/` 資料夾壓縮傳給彙整機：

```bash
# On each machine, compress results
zip -r results_person1.zip results/

# On aggregation machine, unzip all and run
python scripts/plot_comparison.py
python scripts/plot_adaptation.py
python scripts/plot_gru_latent.py
```

---

## 評估指標 / Evaluation Metrics

每個訓練腳本的 `log.csv` 包含以下欄位：

| 欄位 | 說明 |
|---|---|
| `mean_reward` | 每 epoch 平均 reward（= -halting vehicles） |
| `avg_wait` | 每 epoch 平均等待時間（秒，越低越好） |
| `r_morning` | 早峰階段平均 reward |
| `r_off` | 離峰階段平均 reward |
| `r_evening` | 晚峰階段平均 reward |

---

## 突發事件場景 / Sudden Context Shift Scenario

`nets/sudden.rou.xml` 在正常三段之後加入第四段突發車潮：
- **時間**: 10800–12600s（env steps 1080–1259）
- **車流**: EB=698、WB=978、NB=305、SB=338（約正常晚峰的 1.5–2 倍）
- **用途**: 評估各算法對完全未見過的突發事件的鲁棒性

```bash
python scripts/train_fixed_time.py --sudden   # fixed-time on sudden scenario
```

---

## 固定時制基準線 / Fixed-Time Baseline

`scripts/train_fixed_time.py`：60s/60s 固定綠燈週期，不做任何學習。
作為最基本的比較基準（論文中 Expected Results 提到的 Fixed-time control）。

---

## 視覺化腳本 / Visualization Scripts

| 腳本 | 輸出 | 說明 |
|---|---|---|
| `scripts/plot_comparison.py` | `figures/reward_comparison.png` | 整體 reward 比較 |
| `scripts/plot_adaptation.py` | `figures/adaptation_speed.png` | context 切換後適應速度 |
| `scripts/plot_gru_latent.py` | `figures/gru_latent_tsne.png` | GRU h_t t-SNE 分群 |

---

## src/ 目錄結構 / Source Directory

| Path | Description | Source |
|---|---|---|
| `neural_net/nn.py` | FCNPolicy, FullyConnectNN | LCPO disc-gym (verbatim) |
| `neural_net/gru_encoder.py` | GRUEncoder, ObsHistoryEncoder | **新寫 NEW** |
| `buffer/buffer.py` | Replay buffer | LCPO disc-gym (verbatim) |
| `buffer/buffer_ood.py` | OOD reservoir sampler | LCPO windy-gym (verbatim) |
| `agent/core_alg/core_pg.py` | GAE + policy gradient | LCPO disc-gym (verbatim) |
| `agent/core_alg/core_trpo.py` | Conjugate gradient | LCPO disc-gym (verbatim) |
| `agent/core_alg/core_utils.py` | Flat param helpers | LCPO disc-gym (verbatim) |
| `agent/core_alg/core_lcpo.py` | LCPO TRPO update | LCPO windy-gym (verbatim) |
| `agent/core_alg/core_dqn.py` | Double DQN update | LCPO disc-gym (verbatim) |
| `agent/gru_lcpo.py` | GRU-LCPO agent class | **新寫 NEW** |
| `utils/rms.py` | RunningMeanStd | LCPO disc-gym (verbatim) |
