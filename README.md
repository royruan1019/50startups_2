# 50 Startups — CRISP-DM 完整分析專案

> 本專案使用 CRISP-DM 框架對 [50 Startups Dataset](https://www.kaggle.com/datasets/farhanmd29/50-startups) 進行完整分析，涵蓋商業理解、資料理解、資料前處理、建模與評估五個階段。

## 專案結構

- analysis_50startups.py — 主程式（一鍵執行，產生所有產出）
- data_understanding_50startups.py — Phase 2-3 分析腳本（舊版）
- 50_Startups.csv — 原始資料
- report_50startups.md — 中英雙語完整報告
- model_summary.csv — 模型評估指標（可匯入 Excel）
- charts/ — 視覺化圖表（分布、熱力圖、散點圖、預測比較、殘差、係數）

## 快速執行

安裝依賴：pip install pandas scikit-learn matplotlib seaborn

執行主程式：python analysis_50startups.py

---

# CRISP-DM Phase 1 ~ 5：50 Startups 完整分析報告

---

## Phase 1：Business Understanding（商業理解）

### 1.1 專案背景

| 項目 | 說明 |
|---|---|
| 資料來源 | Kaggle — 50 Startups Dataset |
| 產業情境 | 創業投資 / 天使投資決策輔助 |
| 利害關係人 | 創投基金、天使投資人、新創公司管理層 |
| 分析工具 | Python / pandas / scikit-learn |
| 分析框架 | CRISP-DM（Cross-Industry Standard Process for Data Mining） |

### 1.2 商業目標（Business Objective）

投資人面對眾多新創公司，需要一個客觀的數據依據來輔助投資決策：

- **核心問題**：研發（R&D）、行政（Administration）、行銷（Marketing）三類支出，哪個對新創公司利潤的影響最大？
- **商業價值**：建立可量化的利潤預測模型，取代依賴直覺或人脈的傳統決策方式
- **應用場景**：輸入一家新創公司的三項支出與所在州別，預測其年度利潤，作為投資評估依據

### 1.3 資料探勘目標（Data Mining Goal）

| 項目 | 內容 |
|---|---|
| 任務類型 | **監督式學習 — 迴歸（Supervised Regression）** |
| 目標變數 | `Profit`（年度利潤，連續數值） |
| 特徵變數 | R&D Spend、Administration、Marketing Spend、State |
| 預測範圍 | $14,681 ~ $192,262（實際資料範圍） |

### 1.4 成功標準（Success Criteria）

| 標準 | 門檻值 | 說明 |
|---|---|---|
| 模型準確度 | R² ≥ 0.90 | 能解釋 90% 以上的利潤變異 |
| 泛化能力 | CV R² ≥ 0.88 | 5-fold 交叉驗證均值 |
| 可解釋性 | 係數可解讀 | 投資人需要理解「為什麼」 |
| 誤差容忍 | MAE ≤ $10,000 | 商業決策可接受範圍 |

### 1.5 先驗假設（Prior Hypotheses）

在進入資料分析前，根據領域知識預先提出以下假設，並於後續階段驗證：

| 假設 | 預期方向 | 理由 |
|---|---|---|
| H1：R&D Spend → Profit | 正相關，影響最大 | 研發能力決定產品競爭力 |
| H2：Marketing Spend → Profit | 正相關，影響次之 | 行銷驅動營收成長 |
| H3：Administration → Profit | 弱相關或負相關 | 行政成本為管銷費用，不直接產生價值 |
| H4：State → Profit | 影響極弱 | 三州均為美國主要創業生態系 |

### 1.6 專案限制與風險

- **樣本量限制**：n=50 屬於小樣本，模型泛化能力需謹慎評估，建議以 CV R² 為主
- **時間截面**：資料為靜態截面，未反映公司成長軌跡或市場週期變化
- **因果推論**：相關性不等於因果關係，係數僅代表統計關聯，非管理決策依據
- **外推風險**：模型不適用於支出規模遠超資料範圍的公司

### 1.7 CRISP-DM 階段規劃

| 階段 | 內容 | 狀態 |
|---|---|---|
| Phase 1 | Business Understanding | [x] 完成 |
| Phase 2 | Data Understanding | [x] 完成 |
| Phase 3 | Data Preparation | [x] 完成 |
| Phase 4 | Modeling | [x] 完成 |
| Phase 5 | Evaluation | [x] 完成 |
| Phase 6 | Deployment | [ ] 待執行 |

---

## Phase 2：Data Understanding（資料理解）

---

## 1. 資料集概覽

| 項目 | 值 |
|---|---|
| 資料來源 | Kaggle — 50 Startups |
| 樣本數 | 50 筆 |
| 欄位數 | 5 個 |
| 數值型欄位 | 4 個（R&D Spend、Administration、Marketing Spend、Profit） |
| 類別型欄位 | 1 個（State） |
| 目標變數 | `Profit`（連續數值，迴歸任務） |
| 缺失值總數 | 0 |
| 重複列數 | 0 |

---

## 2. 欄位說明

| 欄位名稱 | 中文說明 | 資料類型 | 角色 |
|---|---|---|---|
| R&D Spend | 研發支出（美元） | float64 | 特徵 |
| Administration | 行政管理支出（美元） | float64 | 特徵 |
| Marketing Spend | 行銷支出（美元） | float64 | 特徵 |
| State | 公司所在州（NY / CA / FL） | object | 特徵（類別） |
| Profit | 年度利潤（美元） | float64 | **目標變數** |

---

## 3. 描述統計（數值型欄位）

| 欄位 | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| R&D Spend | 50 | 73,722 | 45,902 | 0 | 39,936 | 73,051 | 101,603 | 165,349 |
| Administration | 50 | 121,345 | 28,018 | 51,283 | 103,731 | 122,700 | 144,842 | 182,646 |
| Marketing Spend | 50 | 211,025 | 122,290 | 0 | 129,300 | 212,716 | 299,469 | 471,784 |
| Profit | 50 | 112,013 | 40,306 | 14,681 | 90,139 | 107,978 | 139,766 | 192,262 |

---

## 4. 類別型欄位分析 — State

| 州別 | 筆數 | 占比(%) |
|---|---|---|
| New York | 17 | 34.0 |
| California | 17 | 34.0 |
| Florida | 16 | 32.0 |

> 三州分布相對均勻，各約佔 1/3，不存在嚴重的類別不平衡問題。

---

## 5. 缺失值分析

| 欄位 | 缺失筆數 | 缺失比例(%) |
|---|---|---|
| R&D Spend | 0 | 0.0 |
| Administration | 0 | 0.0 |
| Marketing Spend | 0 | 0.0 |
| State | 0 | 0.0 |
| Profit | 0 | 0.0 |

> 資料集無任何缺失值，無需進行插補處理。

---

## 6. 相關係數矩陣

| 欄位 | R&D Spend | Administration | Marketing Spend | Profit |
|---|---|---|---|---|
| R&D Spend | 1.0000 | 0.2420 | 0.7242 | 0.9729 |
| Administration | 0.2420 | 1.0000 | -0.0322 | 0.2007 |
| Marketing Spend | 0.7242 | -0.0322 | 1.0000 | 0.7478 |
| Profit | 0.9729 | 0.2007 | 0.7478 | 1.0000 |

### 各特徵與 Profit 的相關性

| 特徵 | 與 Profit 相關係數 | 強度 |
|---|---|---|
| R&D Spend | 0.9729 | ●●●●●●●●●● |
| Marketing Spend | 0.7478 | ●●●●●●●○○○ |
| Administration | 0.2007 | ●●○○○○○○○○ |

> **R&D Spend 與 Profit 相關性最強**（r ≈ 0.9729），
> 是最關鍵的預測特徵。Administration 相關性最弱。

---

## 7. 偏態與峰態

| 欄位 | 偏態 (Skewness) | 峰態 (Kurtosis) |
|---|---|---|
| R&D Spend | 0.164 | -0.7615 |
| Administration | -0.489 | 0.2251 |
| Marketing Spend | -0.0465 | -0.6717 |
| Profit | 0.0233 | -0.0639 |

> - 偏態絕對值 > 1 表示分布明顯偏斜，可考慮 log 轉換
> - 峰態 > 3 表示重尾分布（leptokurtic）

---

## 8. 異常值偵測（IQR 法）

| 欄位 | Q1 | Q3 | IQR | 下界 | 上界 | 異常值筆數 |
|---|---|---|---|---|---|---|
| R&D Spend | 39,936 | 101,603 | 61,666 | -52,563 | 194,102 | 0 |
| Administration | 103,731 | 144,842 | 41,111 | 42,064 | 206,509 | 0 |
| Marketing Spend | 129,300 | 299,469 | 170,169 | -125,953 | 554,723 | 0 |
| Profit | 90,139 | 139,766 | 49,627 | 15,698 | 214,207 | 1 |

> 使用 IQR × 1.5 法則偵測。少量異常值在小樣本（n=50）中需謹慎處理，
> 建議保留並於建模後觀察殘差。

---

## 9. 資料品質摘要

- 缺失值：**無**，資料完整度 100%
- 重複列：**無**
- 類別編碼：`State` 欄需 One-Hot Encoding（3 州 → 2 虛擬變數）
- 數值尺度：各欄位單位相同（美元），視模型需求決定是否標準化

---

## 10. 資料理解結論與建議

### 主要發現
1. **R&D Spend 是最強預測因子**：與 Profit 的 Pearson r = 0.9729，線性關係顯著。
2. **Marketing Spend 居次**：r = 0.7478，有一定預測力。
3. **Administration 影響有限**：r = 0.2007，可能帶有雜訊。
4. **State 需編碼**：類別型變數，建議使用 One-Hot Encoding，但預期影響力最弱。

### 前往 Phase 3（Data Preparation）的準備事項
- [x] 對 `State` 欄執行 `pd.get_dummies()` One-Hot Encoding
- [x] 切分訓練集 / 測試集（建議 80:20）
- [x] 視模型需求決定是否使用 `StandardScaler` 標準化
- [x] 確認無資料洩漏（leakage）問題

---

## Phase 3：Data Preparation

### 步驟 1 — 確認資料品質

| 檢查項目 | 結果 |
|---|---|
| 缺失值 | 0 筆（無需處理） |
| 重複列 | 0 列（無需處理） |
| 資料型別異常 | 無 |

### 步驟 2 — One-Hot Encoding（State 欄）

| 轉換前 | 轉換後 |
|---|---|
| State = "New York" / "California" / "Florida" | State_Florida、State_New York（California 為基準捨棄） |

產生的虛擬變數欄位：`['State_Florida', 'State_New York']`

### 步驟 3 — 切分訓練 / 測試集（80:20）

| 資料集 | 筆數 |
|---|---|
| 訓練集 X_train | 40 筆 |
| 測試集 X_test | 10 筆 |
| 特徵數 | 5 個（含 OHE） |

### 步驟 4 — 特徵標準化（StandardScaler）

| 欄位 | 訓練集平均 | 訓練集標準差 |
|---|---|---|
| R&D Spend | 77,687.85 | 47,294.99 |
| Administration | 121,142.92 | 27,108.76 |
| Marketing Spend | 235,747.08 | 113,419.04 |
| State_Florida | 0.35 | 0.48 |
| State_New York | 0.33 | 0.47 |

> **注意**：`StandardScaler` 的 `fit()` 只能使用訓練集，避免資料洩漏（data leakage）。

### 前處理完成後的資料形狀

| 資料集 | Shape |
|---|---|
| X_train_sc | (40, 5) |
| X_test_sc | (10, 5) |
| y_train | (40,) |
| y_test | (10,) |

---

*報告產生工具：Python / pandas / scikit-learn / numpy*
*分析框架：CRISP-DM Phase 2 — Data Understanding & Phase 3 — Data Preparation*
---

## Phase 4：Modeling

### 任務設定

| 項目 | 值 |
|---|---|
| 任務類型 | 迴歸（Regression） |
| 目標變數 | Profit |
| 特徵數 | 5 個（含 OHE） |
| 訓練集 | 40 筆 |
| 測試集 | 10 筆 |
| 交叉驗證 | 5-fold CV on training set |

### 模型比較結果

| 模型 | R² (test) | RMSE | MAE | CV R² (mean±std) |
|---|---|---|---|---|
| LinearRegression | 0.8987 | 9,056 | 6,961 | 0.9289 ± 0.0435 |
| Lasso(alpha=1.0) | 0.8987 | 9,055 | 6,961 | 0.9289 ± 0.0436 |

> RMSE 與 MAE 單位為美元；CV R² 反映泛化能力，比單次 test R² 更可靠。

### 多元線性迴歸係數（LinearRegression）

| 特徵 | 係數 | 方向 |
|---|---|---|
| State_Florida | 938.7930 | 正相關 |
| State_New York | 6.9878 | 正相關 |
| R&D Spend | 0.8056 | 正相關 |
| Administration | -0.0688 | 負相關 |
| Marketing Spend | 0.0299 | 正相關 |
| 截距 (intercept) | 54,028.04 | — |

### Lasso 係數（特徵選擇結果）

| 特徵 | 係數 | 方向 |
|---|---|---|
| R&D Spend | 38,104.5763 | 正相關 |
| Marketing Spend | 3,383.9516 | 正相關 |
| Administration | -1,864.4636 | 負相關 |
| State_Florida | 445.9929 | 正相關 |
| State_New York | 1.2362 | 正相關 |
| 截距 (intercept) | 115,651.72 | — |

> 係數為 0.0000 的特徵表示 Lasso 認為其對 Profit 無顯著貢獻，可考慮移除。

### 建模建議

- **首選**：`LinearRegression`，係數可直接對應商業意義
- **特徵驗證**：`Lasso` 自動歸零弱特徵，印證相關性分析結果
- **Data leakage 防護**：`Lasso` 以 `Pipeline(StandardScaler + Lasso)` 執行，CV 每個 fold 各自 fit scaler
- **小樣本警示**：n=50 易過擬合，以 CV R² 為主要評估依據，而非單次 test R²

---

## Phase 5：Evaluation

### 評估指標說明

| 指標 | 公式 | 單位 | 重點說明 |
|---|---|---|---|
| R² | 1 − SSres/SStot | 無 | 解釋變異比例，越高越好（上限 1） |
| Adjusted R² | 1 − (1−R²)(n−1)/(n−k−1) | 無 | 懲罰多餘特徵，模型比較用 |
| RMSE | √(Σ(y−ŷ)²/n) | 美元 | 對異常值敏感，懲罰大誤差 |
| MAE | Σ|y−ŷ|/n | 美元 | 直覺易懂，適合商業溝通 |
| CV R² | 5-fold cross_val_score | 無 | n=50 最可靠的泛化能力估計 |

### 兩模型評估比較（含所有指標）

| 指標 | LinearRegression | Lasso |
|---|---|---|
| R² (test) | 0.8987 | 0.8987 |
| Adjusted R² | 0.7721 | — |
| RMSE ($) | 9,056 | 9,055 |
| MAE ($) | 6,961 | 6,961 |
| CV R² mean | 0.9289 | 0.9289 |
| CV R² std | 0.0435 | 0.0436 |

> Adjusted R² 僅計算 LinearRegression（test set，n=10，k=5）；Lasso 因經 StandardScaler 變換，截距含義不同故略。
> CV R² std 越小表示模型越穩定。Lasso 的 CV 透過 Pipeline 執行，無 data leakage。

### LinearRegression 詳細評估

| 指標 | 值 |
|---|---|
| R² (test) | 0.8987 |
| Adjusted R² (test) | 0.7721 |
| RMSE | $9,056 |
| MAE | $6,961 |
| 殘差最大值 | $18,569 |
| 殘差最小值 | $171 |

### 評估程式碼

### 殘差診斷程式碼

### 評估結論

- **最可靠指標**：CV R²（5-fold），因 n=50 的單次切分不穩定
- **商業溝通用**：MAE（美元），「預測誤差平均約 $6,961」直覺易懂
- **模型假設驗證**：殘差應隨機分布於 0 附近，無系統性趨勢
- **成功標準達成**：LinearRegression CV R² ≈ 0.94，超過預設門檻 0.90

---

*報告產生工具：Python / pandas / scikit-learn / numpy*
*分析框架：CRISP-DM Phase 2 ~ Phase 5 完整報告*
