# fndiag 升級規劃:從「六支關聯腳本」到「模型評估框架」

> 這份文件是給 Claude Code 的實作簡報(implementation brief)。目標是把現有的
> `fndiag`(六個各自獨立的診斷步驟)重構成一個模組化、可擴充、輸出格式標準化的
> 模型評估框架,並補上目前完全缺少的「輸入資料品質」與「FP 根因分類」能力。
>
> 執行時請**按照下方「建議實作順序」分階段進行**,每個階段結束後跑一次
> `pytest` 並確認現有六個診斷步驟的輸出數值不變(除非該階段明確要改欄位/檔名),
> 再繼續下一階段。

---

## 0. 背景與現況

`fndiag` 目前的結構:

```
fndiag/
  src/fndiag/
    cli.py                       # argparse 手刻的子指令
    core/
      matching.py                 # GT<->預測 一對一配對(對齊 metrics.py 的 compute_object_metrics)
      io_utils.py                  # 資料夾/檔案探索(對齊 analysis.py 的動態偵測慣例)
      reporting.py                  # --sample/--model/--output-dir 篩選 + 檔名標註
    steps/
      iou_distribution.py         # Step 1
      fn_visualization.py         # Step 2
      fn_characteristics.py       # Step 3
      detection_probability.py    # Step 4
      fn_contribution.py          # Step 5
      story_closure.py            # Step 6
  tests/
    test_matching.py
    test_io_utils_and_reporting.py
```

`core/matching.py` 已經把 TP/FN/FP 的判定邏輯改成跟實驗室內部 `metrics.py` 的
`compute_object_metrics` 完全一致的一對一貪婪配對演算法(同樣的迭代順序、同樣
0.5 IoU 門檻),並新增了 `find_false_positives()`。這部分**已經完成,不需要重做**,
升級時要繼續沿用。

### 現況的結構性問題

1. **沒有共用的資料層**:每個 step 各自重新掃資料夾、重新讀 TIFF、重新跑
   `match_instances`,同一組 GT/預測配對可能被重複計算多次,沒有中間結果可以
   跨 step 重複利用。
2. **輸出格式不統一**:六個 step 各自 hardcode 自己的 DataFrame 欄位、自己存
   PNG,沒有共同的「結果表」schema,無法跨 step 做進一步分析(例如「FP 多的
   樣本是不是 raw image 對比度也比較差」目前做不到)。
3. **只評估「模型輸出 vs GT」,沒有評估「輸入資料本身」**:Raw image 品質
   (雜訊、對比度、飽和/欠曝、模糊)、GT 標註品質(離群值、邊界截斷、標註者
   間一致性)完全沒有涵蓋,但這往往才是問題根源。
4. **FP/FN 沒有做根因分類**:現在只知道「這是一個 FP」,不知道它是「雜訊誤判」
   「邊界雙重標記(過度分割)」還是「模型幻覺出一顆看起來很真的假細胞」——
   這三種問題的處理方式完全不同。
5. **工程慣例**:argparse 手刻子指令、無 CI、無型別檢查、無 pre-commit,跟
   同領域主流工具(napari、cellpose、stardist)比起來還缺一截。

---

## 1. 目標架構總覽

把現在「讀圖 → 算指標 → 畫圖 → 存檔」全部黏在一起的六支腳本,拆成兩個階段:

```
┌─────────────────┐     ┌──────────────────────────────┐     ┌────────────────────┐
│  1. Collect      │ --> │  2. Checks (analysis + report) │ --> │  3. Writers (輸出)   │
│  掃資料夾、讀圖、  │     │  查詢 collect 產出的表,       │     │  csv / parquet /    │
│  跑配對演算法、    │     │  做統計/畫圖,回傳標準化的     │     │  html / png 統一寫檔 │
│  寫成標準化長表    │     │  ReportArtifact               │     │                     │
└─────────────────┘     └──────────────────────────────┘     └────────────────────┘
```

- **Collect 只做一次**,產出快取(parquet),之後所有分析都是查表,不重新讀 TIFF。
- **Checks 是可插拔的**(plugin-style registry),現有六個 step 改寫成 Check,
  再新增三個沒有的能力(raw image 品質、GT 標註品質、FP 根因分類)。
- **輸出一律走 `ReportArtifact` → `ReportWriter`**,格式(csv/parquet/html/xlsx)
  由 CLI flag 決定,不是每個 check 自己決定要存什麼檔名/格式。

---

## 2. 新增/修改的檔案結構

```
fndiag/
  src/fndiag/
    cli.py                          # 改用 typer,子指令改吃 checks registry

    core/
      matching.py                   # 不動(既有一對一配對演算法保留)
      io_utils.py                   # 不動
      reporting.py                  # 不動,但輸出路徑邏輯移交給 writers.py

      schema.py                     # 新增:InstanceRecord / ImageQualityRecord dataclass
      pipeline.py                   # 新增:collect() 主流程,產出標準化長表 + parquet 快取
      report.py                     # 新增:ReportArtifact dataclass
      writers.py                    # 新增:CsvWriter / ParquetWriter / HtmlWriter / (Xlsx 可選)
      config.py                     # 新增:fndiag.toml 設定檔載入(pydantic-settings 或 tomllib)

    checks/                         # 取代 steps/ 目錄
      __init__.py                   # CHECKS registry: name -> Check 實例
      base.py                       # Check 抽象基底類別

      iou_distribution.py           # 原 Step 1,改吃 collect 產出的表
      fn_visualization.py           # 原 Step 2(仍需直接讀圖做 3D 裁切,無法完全脫離 I/O)
      fn_characteristics.py         # 原 Step 3
      detection_probability.py      # 原 Step 4
      fn_contribution.py            # 原 Step 5
      story_closure.py              # 原 Step 6

      raw_image_quality.py          # 新增:SNR / 模糊 / 飽和 / 對比度
      gt_annotation_quality.py      # 新增:GT 離群值 / 邊界截斷 / 密度異常
      fp_root_cause.py              # 新增:FP 細分類(noise / boundary_split / hallucination)

  tests/
    test_matching.py                # 不動
    test_io_utils_and_reporting.py  # 不動
    test_schema.py                  # 新增
    test_pipeline.py                # 新增
    test_checks/                    # 新增:每個 check 一支測試檔
      test_raw_image_quality.py
      test_gt_annotation_quality.py
      test_fp_root_cause.py

  .github/
    workflows/ci.yml                # 新增:pytest + ruff + mypy,矩陣測 3.9~3.12
    ISSUE_TEMPLATE/
    PULL_REQUEST_TEMPLATE.md

  .pre-commit-config.yaml           # 新增:ruff + black + mypy
  docs/                             # 新增:mkdocs 文件(取代/補充 README)
  mkdocs.yml
  fndiag.toml.example                # 新增:設定檔範例
  pyproject.toml                    # 更新:加 typer/pydantic/pyarrow/mypy/pytest-cov 等
```

---

## 3. 核心資料層:`core/schema.py`

這是所有其他改動的地基。定義兩張標準表(long format,類似 tidy data / dbt 的概念):

```python
# fndiag/core/schema.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class InstanceRecord:
    """One row = one GT or predicted instance.

    This is the atomic unit every downstream check (FN characteristics, ROI,
    detection curves, FP root-cause...) is derived from. Nothing re-reads
    TIFFs after this table exists.
    """
    dataset: str
    sample: str
    model: str
    slice_name: str
    z_index: int
    role: str              # "gt" | "prediction"
    instance_id: int
    volume: int
    mean_intensity: Optional[float]
    centroid_y: float
    centroid_x: float
    bbox_min_row: int
    bbox_min_col: int
    bbox_max_row: int
    bbox_max_col: int
    classification: str    # "true_positive" | "blind_fn" | "merged_fn" | "false_positive"
    best_iou: Optional[float]
    matched_instance_id: Optional[int]
    fp_subtype: Optional[str] = None   # 由 fp_root_cause check 事後補上


@dataclass(frozen=True)
class ImageQualityRecord:
    """One row = one raw/GT slice's image-quality metrics."""
    dataset: str
    sample: str
    slice_name: str
    z_index: int
    source: str             # "raw" | "gt_mask" | "prediction"
    mean_intensity: float
    std_intensity: float
    snr_estimate: float
    saturation_pct: float             # % pixels at max dtype value
    laplacian_variance: float         # focus/blur proxy
    gt_instance_count: int
    gt_touching_border_pct: float     # 可能被截斷的標註
```

實作要點:

- 用 `dataclasses.asdict()` 或 `pandas.DataFrame.from_records()` 轉成 DataFrame。
- 兩張表都用 **long format**(一列一個觀測值),而不是像現在每個 step 各自
  pivot 成寬表——寬表留給畫圖前的最後一步再做,不要污染儲存層。
- 欄位命名沿用既有的 snake_case 慣例(`volume`、`mean_intensity`、`classification`
  這些已經在上一輪重構定案,不要再改)。

---

## 4. 執行引擎:`core/pipeline.py`

```python
# fndiag/core/pipeline.py
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

from fndiag.core.io_utils import find_gt_folders, find_pred_folders, resolve_image_dir, extract_model_name
from fndiag.core.matching import match_instances, find_false_positives
from fndiag.core.schema import InstanceRecord, ImageQualityRecord


def collect(
    root: Path,
    *,
    sample_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single pass over the dataset: for every GT/prediction/raw triplet,
    run match_instances + find_false_positives (+ image-quality checks),
    and return (instances_df, quality_df).

    If cache_path is given and exists (and force_refresh is False), load
    from cache instead of re-scanning. Every check downstream should read
    these DataFrames instead of touching TIFFs directly.
    """
    if cache_path and cache_path.exists() and not force_refresh:
        return (
            pd.read_parquet(cache_path.with_suffix(".instances.parquet")),
            pd.read_parquet(cache_path.with_suffix(".quality.parquet")),
        )

    instance_rows: list[InstanceRecord] = []
    quality_rows: list[ImageQualityRecord] = []

    # ... 掃描邏輯沿用 io_utils.find_gt_folders / find_pred_folders /
    #     resolve_image_dir / extract_model_name,對每個 slice 呼叫
    #     match_instances() + find_false_positives(),並呼叫
    #     checks/raw_image_quality.py 裡抽出來的純函式算 ImageQualityRecord。

    instances_df = pd.DataFrame([r.__dict__ for r in instance_rows])
    quality_df = pd.DataFrame([r.__dict__ for r in quality_rows])

    if cache_path:
        instances_df.to_parquet(cache_path.with_suffix(".instances.parquet"))
        quality_df.to_parquet(cache_path.with_suffix(".quality.parquet"))

    return instances_df, quality_df
```

實作要點:

- **快取用 parquet**(需要 `pyarrow`,加進 `pyproject.toml` 依賴)。
- `force_refresh` 給 CLI 一個 `--refresh-cache` flag,方便重跑。
- collect 階段要把 `fn_visualization.py`(Step 2)以外的所有計算邏輯都搬進來
  ——Step 2 因為需要動態裁切 3D Z-context 圖片、且是抽樣視覺化而非統計匯總,
  維持獨立直接讀圖即可,不用勉強塞進這張長表。

---

## 5. 標準化輸出:`core/report.py` + `core/writers.py`

```python
# fndiag/core/report.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

@dataclass
class ReportArtifact:
    """Every check returns a list of these. One artifact = one table
    (always) + optional one figure. This is the single format every
    writer (csv/parquet/html/xlsx) knows how to serialize."""
    name: str                        # e.g. "step3_fn_characteristics"
    table: pd.DataFrame
    figure: Optional["matplotlib.figure.Figure"] = None
    metadata: dict = field(default_factory=dict)  # 篩選條件、使用的門檻值等
```

```python
# fndiag/core/writers.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from fndiag.core.report import ReportArtifact

class ReportWriter(ABC):
    extension: str

    @abstractmethod
    def write(self, artifact: ReportArtifact, output_dir: Path) -> Path: ...

class CsvWriter(ReportWriter):
    extension = ".csv"
    def write(self, artifact, output_dir):
        path = output_dir / f"{artifact.name}{self.extension}"
        artifact.table.to_csv(path, index=False)
        if artifact.figure is not None:
            artifact.figure.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
        return path

class ParquetWriter(ReportWriter):
    extension = ".parquet"
    def write(self, artifact, output_dir):
        path = output_dir / f"{artifact.name}{self.extension}"
        artifact.table.to_parquet(path, index=False)
        return path

class HtmlWriter(ReportWriter):
    """產生一個可互動的總覽頁面,整合多個 artifact 的表 + 圖。"""
    extension = ".html"
    ...

WRITER_REGISTRY: dict[str, type[ReportWriter]] = {
    "csv": CsvWriter,
    "parquet": ParquetWriter,
    "html": HtmlWriter,
}
```

CLI 新增 `--format csv,parquet,html`(逗號分隔,可多選),取代現在每個 step
自己決定「存 png」的寫死行為。

---

## 6. Checks 架構:`checks/base.py` + registry

```python
# fndiag/checks/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import argparse
import pandas as pd
from fndiag.core.report import ReportArtifact

class Check(ABC):
    """Every check is: given the collected instance/quality tables, produce
    zero or more standardized ReportArtifact objects."""

    name: str
    description: str

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    @abstractmethod
    def run(
        self,
        instances: pd.DataFrame,
        quality: pd.DataFrame,
        args: argparse.Namespace,
    ) -> list[ReportArtifact]: ...
```

```python
# fndiag/checks/__init__.py
from fndiag.checks.iou_distribution import IouDistributionCheck
from fndiag.checks.fn_characteristics import FnCharacteristicsCheck
from fndiag.checks.detection_probability import DetectionProbabilityCheck
from fndiag.checks.fn_contribution import FnContributionCheck
from fndiag.checks.story_closure import StoryClosureCheck
from fndiag.checks.raw_image_quality import RawImageQualityCheck
from fndiag.checks.gt_annotation_quality import GtAnnotationQualityCheck
from fndiag.checks.fp_root_cause import FpRootCauseCheck

CHECKS: dict[str, Check] = {
    c.name: c for c in [
        IouDistributionCheck(),
        FnCharacteristicsCheck(),
        DetectionProbabilityCheck(),
        FnContributionCheck(),
        StoryClosureCheck(),
        RawImageQualityCheck(),
        GtAnnotationQualityCheck(),
        FpRootCauseCheck(),
    ]
}
```

新增 check 只需要:繼承 `Check`、實作 `run()` 回傳 `list[ReportArtifact]`、在
`CHECKS` 註冊一行——不需要碰任何 I/O 或 CLI 邏輯。

---

## 7. 三個新增的品質檢查(具體邏輯)

### 7.1 `checks/raw_image_quality.py` ——輸入影像品質

回答:「recall 低是模型的問題,還是原始影像本身品質就差?」

- **SNR 估計**:前景(GT 涵蓋範圍)平均強度 vs 背景(GT 以外區域)標準差的比值
- **模糊偵測**:對 raw image 做 Laplacian,取變異數(`laplacian_variance` 越低越模糊)
  ```python
  from scipy.ndimage import laplace
  laplacian_variance = float(laplace(raw_arr.astype(float)).var())
  ```
- **飽和/欠曝比例**:`(raw_arr == raw_arr.max()).mean()` 之類的統計
- **對比度**:直方圖是否雙峰(前景/背景分離良好)vs 單峰糊成一團

輸出一張表(每 sample 一列的彙總 + 每 slice 一列的明細),外加一張「品質分數
vs 該樣本的 recall/FP 率」散佈圖——直接回答「品質差的樣本是不是剛好也是
model 表現差的樣本」。

### 7.2 `checks/gt_annotation_quality.py` ——GT 標註品質

回答:「這批 GT 本身有沒有問題?」

- **離群值偵測**:GT 細胞體積分布用 IQR 或 z-score 抓離群值,列出可疑細胞
  (太大可能是沾黏、太小可能是雜訊誤標)
- **邊界截斷偵測**:`bbox` 是否貼齊影像邊界(`min_row==0` 或 `max_row==shape[0]`
  等),這類細胞的體積/形狀是不完整的,不該被當成正常樣本統計
- **密度異常偵測**:同一個 dataset 底下,不同 sample/slice 之間 GT 密度
  (每張 slice 幾顆細胞)是否有斷崖式變化——可能代表標註者換人、標註標準不一致

### 7.3 `checks/fp_root_cause.py` ——FP 根因分類

在 collect 階段算出的每個 FP,補上 `fp_subtype` 欄位(規則式分類,之後也可以
換成小型分類器):

```python
def classify_fp_subtype(fp_volume, fp_intensity, nearest_gt_distance, background_intensity_stats):
    if fp_volume < SMALL_VOLUME_THRESHOLD and fp_intensity ~= background:
        return "noise_fp"
    if nearest_gt_distance < BOUNDARY_SPLIT_DISTANCE_THRESHOLD:
        return "boundary_split_fp"      # 疑似過度分割,同一顆細胞被切成兩塊
    return "hallucination_fp"           # 大小/亮度都像真細胞,但完全沒有 GT 對應,最值得關注
```

輸出:每個 subtype 的數量、體積/亮度分布,以及「hallucination_fp 是否集中在
特定 sample/model」的交叉表——這是目前 fndiag 完全沒有的能力,也是使用者
最早提到「診斷過度分割/幻覺問題」的直接落地。

---

## 8. CLI 改用 `typer`

```python
# fndiag/cli.py (節錄)
import typer
from fndiag.checks import CHECKS
from fndiag.core.pipeline import collect
from fndiag.core.writers import WRITER_REGISTRY

app = typer.Typer(help="Diagnostic toolkit for false-negative analysis of 3D instance segmentation models.")

@app.command()
def run(
    check: str = typer.Argument(..., help=f"One of: {', '.join(CHECKS)}, or 'all'"),
    base_dir: Path = typer.Option(..., "--base-dir"),
    sample: Optional[str] = typer.Option(None, "--sample"),
    model: Optional[str] = typer.Option(None, "--model"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    format: str = typer.Option("csv", "--format", help="csv,parquet,html"),
    refresh_cache: bool = typer.Option(False, "--refresh-cache"),
):
    instances, quality = collect(base_dir, sample_filter=sample, model_filter=model, force_refresh=refresh_cache)
    checks_to_run = CHECKS.values() if check == "all" else [CHECKS[check]]
    for c in checks_to_run:
        for artifact in c.run(instances, quality, ...):
            for fmt in format.split(","):
                WRITER_REGISTRY[fmt]().write(artifact, output_dir or base_dir)
```

`typer` 比手刻 argparse 好在:自動產生漂亮的 `--help`、型別提示直接變成 CLI
驗證、子指令定義更精簡、原生支援 `Path`/`Optional` 型別。

---

## 9. 設定檔支援:`fndiag.toml`

取代現在一長串 CLI flags,允許專案根目錄放設定檔:

```toml
# fndiag.toml
[dataset]
raw_name = "Flatten_561"
dark_name = "Flatten_561_dark"
mask_name = "Flatten_561_mask"

[thresholds]
tp_iou = 0.5
blind_fn_iou = 0.05
fp_boundary_split_distance = 5.0

[output]
format = ["csv", "html"]
output_dir = "results/"
```

用 `pydantic-settings` 或標準庫 `tomllib`(Python 3.11+ 內建,3.9/3.10 用
`tomli` backport)載入,CLI flag 優先權高於設定檔。

---

## 10. 工程慣例對齊主流 GitHub 專案

| 項目 | 現況 | 建議 |
|---|---|---|
| CLI 框架 | 手刻 argparse | 改用 `typer` |
| 型別檢查 | 無 | 加 `mypy`,`pyproject.toml` 內 `[tool.mypy]`,CI 強制跑 |
| Lint/format | `ruff` 在 dev extra 但無強制 | 加 `.pre-commit-config.yaml`(ruff + black + mypy) |
| CI | 無 | `.github/workflows/ci.yml`:pytest + ruff + mypy,矩陣測 3.9~3.12 |
| 測試覆蓋率 | 有 pytest,無門檻 | 加 `pytest-cov` + CI 覆蓋率門檻(建議 >80%) |
| 版本管理 | 手動改 `__version__` | 改用 `setuptools-scm`,從 git tag 自動產生版本號 |
| 文件 | 只有 README | 加 `mkdocs` + `mkdocs-material`,發布到 GitHub Pages |
| 設定檔 | 只有 CLI flags | 見上方第 9 節 `fndiag.toml` |
| Logging | `logging.basicConfig` | 改用 `rich.logging.RichHandler`,取代 tqdm 進度條 |
| Issue/PR 模板 | 無 | `.github/ISSUE_TEMPLATE/`、`PULL_REQUEST_TEMPLATE.md` |
| 快取/中繼資料 | 無 | 見上方 `core/pipeline.py` 的 parquet 快取 |

---

## 11. 建議實作順序(給 Claude Code 的執行計畫)

請依序完成以下階段,**每個階段結束後跑 `pytest` 確認測試通過**,並在
`CHANGELOG.md` 加一筆記錄再進下一階段。除非該階段明確要求改變輸出,否則
既有六個 check 的輸出數值必須跟重構前一致(可以用第二次對話中建立的合成
資料集回歸測試)。

### 階段一:資料層地基(不改變任何現有輸出)
1. 新增 `core/schema.py`(`InstanceRecord` / `ImageQualityRecord`)
2. 新增 `core/pipeline.py` 的 `collect()`,把現有六個 step 內重複的
   「掃資料夾 → 讀圖 → match_instances → find_false_positives」邏輯統一搬進來
3. 新增 `tests/test_schema.py`、`tests/test_pipeline.py`
4. 確認:對同一份合成資料集,`collect()` 產出的表跟現有六個 step 各自算出來
   的數字完全一致(寫一支交叉驗證測試)

### 階段二:輸出標準化 + Check 抽象層
1. 新增 `core/report.py`(`ReportArtifact`)、`core/writers.py`
   (`CsvWriter`/`ParquetWriter`/`HtmlWriter`)
2. 新增 `checks/base.py`(`Check` 抽象類別)
3. 把 `steps/*.py` 搬到 `checks/*.py`,改寫成繼承 `Check`,`run()` 改吃
   `collect()` 產出的表,回傳 `ReportArtifact`,而不是自己存檔
4. `cli.py` 改用 `typer`,子指令改成讀 `CHECKS` registry
5. 舊的 `steps/` 目錄清空,`fndiag.core.reporting` 裡跟檔名標註相關的邏輯
   移進 `writers.py`

### 階段三:三個新增的品質檢查
1. `checks/raw_image_quality.py`
2. `checks/gt_annotation_quality.py`
3. `checks/fp_root_cause.py`(需要先在 `core/pipeline.py` 的 collect 階段
   補上 `fp_subtype` 欄位)
4. 每個都要有對應的 `tests/test_checks/test_*.py`

### 階段四:設定檔 + 工程慣例
1. `core/config.py` + `fndiag.toml.example`
2. `.pre-commit-config.yaml`、`.github/workflows/ci.yml`
3. `pytest-cov`、`setuptools-scm`
4. `mkdocs.yml` + `docs/`(可以直接把現有 README 內容拆分過去)
5. Logging 改用 `rich`

### 階段五:文件收尾
1. 更新 `README.md` 反映新架構(collect/check/writer 三層式說明)
2. `CHANGELOG.md` 補上完整的架構重構記錄(**這會是一次 breaking change**,
   建議版本號跳到 `1.0.0`,並在 CHANGELOG 開頭註明遷移指南:舊的
   `fndiag <step-name> --base-dir ...` 用法 vs 新的
   `fndiag run <check-name> --base-dir ...` 用法對照表)

---

## 12. 驗收標準

- [ ] 對同一份資料集,新架構跑出來的每個 check 數值跟重構前完全一致
      (`tests/` 裡要有交叉驗證)
- [ ] `collect()` 只需要跑一次,後續多個 check 共用同一份快取,不重複讀 TIFF
- [ ] 新增一個 check 不需要碰任何既有檔案,只需要新增一支檔案 + 在
      `CHECKS` registry 註冊一行
- [ ] 所有輸出表都是標準化 schema(至少有 `dataset`/`sample`/`model` 這幾個
      共同欄位可以 join)
- [ ] `raw_image_quality`、`gt_annotation_quality`、`fp_root_cause` 三個新
      check 都能在合成資料集上跑出合理結果
- [ ] CI 綠燈:pytest + ruff + mypy 全過
- [ ] `fndiag --help` / `fndiag run --help` 輸出清楚易讀(typer 自動產生)
