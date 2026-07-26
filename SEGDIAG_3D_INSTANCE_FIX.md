# segdiag：物件計數需要改成「3D 整個 volume 一次標記」，目前是「逐 2D 切片標記」

## 背景

`Chulab-Signal_Segmentation-main/analysis.py` 之前有一個已修好的 bug：`evaluate_triplet()`
原本是在讀每一個 2D z-slice 檔案的迴圈裡，就對那一張 slice 呼叫一次
`skimage.measure.label()` 去標記物件、計數——這會讓一顆貫穿多層 z-slice 的細胞，在每一層
都被算成一個新物件，導致 `gt_count`/`pr_count` 這類「總共有幾顆細胞」的數字被嚴重高估（實測
約 3.6–7.4 倍）。修法是先把所有 slice 疊成一個 3D volume（`np.stack`），再對整個 3D volume
呼叫一次 `label()`，讓同一顆跨多層的細胞只被標記、計數一次。這個修正目前已經在
`analysis.py` 裡（見 `evaluate_triplet()` 裡 `g_slices`/`p_slices` 累積後
`np.stack(..., axis=0)` 再呼叫 `compute_object_metrics()` 的那段）。

**這次比對發現：`segdiag` 從來沒有做過同樣的修正，它的物件計數邏輯完全是逐 2D slice 在做，
跟 `analysis.py` 修好之前是一模一樣的邏輯。**

## 證據

1. **程式碼層面**：`src/segdiag/core/pipeline.py` 的 `collect()` 函式，讀取 GT/預測檔案的
   方式是 `for z_index, gtf in enumerate(gt_files): ... tifffile.imread(...)`，每讀一張
   2D slice，就直接呼叫 `_slice_instance_rows(gt_arr, pr_arr, ...)`（第 311-340 行），而
   `_slice_instance_rows()` 內部呼叫的 `match_and_find_false_positives(gt_arr, pr_arr, ...)`
   （`core/matching.py`）對傳進來的陣列做 `skimage.measure.label(gt_arr > 0, ...)`——
   因為傳進去的 `gt_arr`/`pr_arr` 永遠是單一 2D slice（`tifffile.imread()` 讀一個檔案
   = 一張切片），這就是逐切片標記，不是 3D volume 標記。

2. **數字層面**：這次上傳的 v12 診斷資料（`v12_diag.zip`）裡，`testing-data_v12` 的
   sample 01（GT count = 541）跟同一批資料用修好版 `analysis.py` 算出來的 Case 1
   （GT count = 141）對不上——541 / 141 ≈ 3.84，剛好落在已知的 3.6–7.4 倍膨脹範圍內，
   幾乎可以確定就是同一個 2D/3D bug。

3. **好消息**：`core/matching.py` 裡 `match_instances()`／`find_false_positives()`／
   `match_and_find_false_positives()` 這三個函式本身**是形狀無關的**（shape-agnostic）——
   它們沒有任何假設輸入一定是 2D，`skimage.measure.label()` 傳入 3D 陣列時，`connectivity`
   預設值（`None`）也會自動套用 3D 的滿連通（26-connected），行為是對的。**代表這三個
   核心函式完全不用改，問題全部出在 `pipeline.py` 的 `collect()` 怎麼呼叫它們——現在是
   「每張 2D slice 呼叫一次」，需要改成「每個 (sample, model) 把所有 slice 疊成 3D volume
   後，只呼叫一次」。**

## 需要怎麼改

### 核心改動：`core/pipeline.py` 的 `collect()`

目前邏輯（簡化）：

```python
for z_index, gtf in enumerate(gt_files):
    gt_arr = tifffile.imread(str(gtf))          # 單張 2D slice
    ...
    for p_dir in pred_folders:
        for z_index, gtf in enumerate(gt_files):
            pr_arr = tifffile.imread(str(prf))   # 單張 2D slice
            instance_rows.extend(
                _slice_instance_rows(gt_arr, pr_arr, raw_arrays.get(gtf.name), ...)
            )
```

需要改成：每個 (sample, model) 組合，把該 sample 底下所有 z-slice 的 GT／預測／raw
先疊成 3D array（依檔名排序，跟 `analysis.py` 的 `np.stack(g_slices, axis=0)` 做法一致），
再對整個 3D volume 呼叫**一次** `match_and_find_false_positives()`，而不是每張 slice 各呼叫
一次。大致方向：

```python
gt_vol = np.stack([gt_arrays[f.name] for f in gt_files if f.name in gt_arrays], axis=0)
pr_vol = np.stack([pr_arrays[f.name] for f in gt_files if f.name in pr_arrays], axis=0)
raw_vol = np.stack([raw_arrays[f.name] for f in gt_files if raw_arrays.get(f.name) is not None], axis=0) \
    if all raw slices available else None

matches, false_positives, pairs = match_and_find_false_positives(
    gt_vol, pr_vol, raw_arr=raw_vol, connectivity=connectivity
)
```

`gt_arrays`/`raw_arrays` 這兩個 dict 目前已經存在（第 253-254 行），可以直接重複利用，
只是現在要把整批疊起來一次送進 matching，不要在讀取當下就逐張送進去。

### `z_index` 的意義要跟著改，這是最容易漏掉的地方

`InstanceMatch`/`FalsePositive` 目前的 `centroid`/`bbox` 是 2D 的（因為輸入是 2D slice）；
改成 3D volume 輸入之後，`centroid`/`bbox` 會自動變成 3D（`regionprops` 在 3D 陣列上跑，
回傳的 `centroid`/`bbox` 本來就會多一個 z 維度，不需要額外處理）。

但 `pipeline.py` 目前是「每張 slice 讀出一組 instance_rows，順手把當下的 `z_index`／
`slice_name` 貼上去」——3D 化之後，一個 instance 只會出現「一次」（不再是每張 slice 各一
筆），原本「這一筆屬於哪張 slice」的欄位意義要改成「這個 3D 物件的 z 座標（例如用 3D
centroid 的第一個維度，或 3D bbox 的 z 範圍）」。這會影響下游好幾個 check：

- `checks/gt_annotation_quality.py` 第 125 行：
  `gt.groupby(["sample", "slice_name", "z_index"]).size().rename("gt_count")` ——這是用
  「每個 z_index 底下有幾筆 instance row」畫出 GT 密度隨 Z 軸變化的圖。3D 化之後，
  如果 `z_index` 改成「這個 3D 物件 centroid 的 z 座標（四捨五入到整數）」，這張圖依然
  可以畫（代表「在這個 z 附近有幾顆細胞的中心點」），只是意義從「這張 2D slice 的
  cross-section 數量」變成「中心點落在這個 z 附近的 3D 物件數量」——這是更正確的
  「密度」定義，但曲線形狀可能會變，需要重新檢視。
- `checks/story_closure.py`、`checks/fn_characteristics.py`、`checks/detection_probability.py`、
  `checks/fn_contribution.py`、`checks/fp_root_cause.py`、`checks/iou_distribution.py`、
  `checks/fp_visualization.py`：這幾個檔案都消費 `instances_df`（也就是 `collect()` 回傳的
  instance rows），主要邏輯是依 `classification`／`volume`／`mean_intensity`／`best_iou`
  分組統計，這些欄位在 3D 化之後語意不變（`volume` 現在是 3D 體積、`mean_intensity` 是
  整個 3D 物件的平均亮度），**只要這些檔案沒有隱含假設「一個 gt_id 只會出現在一張
  slice」，邏輯應該不用大改，但仍建議逐一檢查有沒有依賴 `z_index`／`slice_name` 做
  groupby 的地方，確認 3D 化後語意還合理。**

### 不需要動的部分

- `_slice_quality_row()`（`pipeline.py` 第 145 行附近）跟它產生的 `quality_rows`／
  `raw_image_quality.py`：這是在算每一張 2D slice 本身的影像品質（SNR、Laplacian
  variance 等），這些本來就是合理的逐切片量測，跟物件計數無關，**不要改**。
- `core/matching.py` 的三個核心函式（`match_instances`／`find_false_positives`／
  `match_and_find_false_positives`）：如上面所說，這幾個函式是形狀無關的，3D 陣列
  丟進去就會正確運作，**不需要改動函式本體**，只需要改呼叫端（`pipeline.py`）餵給它們
  的是 3D volume 而不是 2D slice。

## 驗證方式

比照 `analysis.py` 當初的驗證邏輯，建議寫一個合成測試：造一個跨 4 層 z-slice 的假細胞
（例如 `gt[1:5, 5:10, 5:10] = 1`，其餘背景為 0），分別跑舊邏輯（逐 slice label）跟新邏輯
（3D volume label 一次），預期舊邏輯算出 4 個 instance、新邏輯算出 1 個。`tests/` 資料夾
底下已經有 `test_matching.py`／`test_pipeline.py`，建議在這兩個檔案裡加上這個案例作為
regression test，避免以後又不小心退回逐切片的寫法。

修好之後，建議重新對 `testing-data_v12` 全部 8 個模型跑一次 `collect()`，確認 sample 01
的 gt_count 從 541 修正到接近 141（跟 `analysis.py` 修好版的 Case 1 對齊），並確認其他
7 個模型／樣本的 gt_count 也都往下修正到合理範圍，作為修復是否成功的量化檢查點。

## 影響範圍提醒

這個 bug 影響的是**所有跟「總共有幾個物件」有關的數字**（`gt_count`／`pr_count`／
`fp_root_cause_summary` 裡各 FP 子類型的 `count`／`recall` by volume bin 的分母分子等），
數量級預期會全面往下修正（約 1/4 到 1/7）。**不影響**純粹逐切片的影像品質量測
（`raw_image_quality`）。修好後，之前基於舊數字做的判讀（例如「sample 03 的 GT 密度
異常」「hallucination_fp 佔比最高」這類結論）建議重新檢視是否還成立——雖然這些多半是
「相對比較」（跨 sample、跨 model 的排序），受絕對數值膨脹的影響可能不大，但既然有
機會用更正確的數字重新驗證，建議修好後全部重跑一次，不要只信任舊數字時代的相對排序。
