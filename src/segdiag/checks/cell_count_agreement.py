"""Cell-count agreement between GT and predictions - MARS 對齊的位置寬鬆版
計數比對(見 ``SEGDIAG_MARS_ALIGNMENT_COMPLETE.md`` Part 3)。

單純比較 ``gt_count`` vs. ``pr_count`` 的總數(``count_tp = min(gt_count,
pr_count)``)完全不檢查預測有沒有真的碰到 GT——一個把預測隨機灑在背景空白
處的模型,只要總數剛好對上,一樣會拿到接近 1.0 的分數。這裡沿用既有的
一對一貪婪配對演算法(:mod:`segdiag.core.matching`),只是把 IoU 門檻從
``TP_IOU_THRESHOLD``(0.5,``obj_f1`` 用的嚴格形狀吻合度)放寬到
``LOCATED_IOU_THRESHOLD``(0.05,只要求「有真的碰到東西」)——這個寬鬆配對
已經由 :func:`segdiag.core.pipeline._volume_instance_rows` 算好,存在每一列
的 ``located_matched`` 欄位裡,這個 check 只需要聚合、不需要重新配對。
"""

from __future__ import annotations

import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from segdiag.checks.base import Check
from segdiag.core.report import ReportArtifact

logger = logging.getLogger(__name__)

_GROUP_KEYS = ["dataset", "sample", "model"]


def _safe_f1(precision: float, recall: float) -> float:
    """F1 from precision/recall, with the same 0/0 convention as
    ``sklearn.metrics.f1_score``: exactly-zero precision *and* recall (a
    model whose predictions never touch any real GT cell) scores ``0.0``,
    not an undefined ``NaN`` from a literal ``0/0`` - NaN is reserved for
    when precision/recall themselves couldn't be computed at all (e.g. no
    predictions/no GT for that group).
    """
    if pd.isna(precision) or pd.isna(recall):
        return float("nan")
    denom = precision + recall
    return 2 * precision * recall / denom if denom else 0.0


class CellCountAgreementCheck(Check):
    name = "cell-count-agreement"
    description = (
        "GT vs. 預測計數一致性:位置寬鬆版 located_count_f1(0.05 IoU 一對一配對)"
        " + Bland-Altman 一致性分析"
    )

    def run(
        self, instances: pd.DataFrame, quality: pd.DataFrame, args: argparse.Namespace
    ) -> List[ReportArtifact]:
        if instances.empty:
            logger.warning("No collected instances - nothing to report.")
            return []

        gt = instances[instances["role"] == "gt"]
        pred = instances[instances["role"] == "prediction"]

        gt_count = gt.groupby(_GROUP_KEYS).size().rename("gt_count")
        tp_count = gt[gt["classification"] == "true_positive"].groupby(_GROUP_KEYS).size()
        fp_count = pred.groupby(_GROUP_KEYS).size()
        pr_count = tp_count.add(fp_count, fill_value=0).rename("pr_count")

        # located_tp: GT 側被寬鬆配對(IoU>=0.05)claim 到的數量 - 這是
        # located_precision/located_recall 共用的分子,跟 classification 是
        # true_positive/merged_fn/blind_fn 完全獨立的第二套配對結果。
        located_tp = gt[gt["located_matched"]].groupby(_GROUP_KEYS).size().rename("located_tp")
        # located_pr_count: 「不是純雜訊/純幻覺」的預測數 - 已經是嚴格 TP 的
        # 預測(一定碰到東西),加上原本被判為 false_positive、但寬鬆配對下仍然
        # 碰到某個 GT 的預測(located_matched=True)。
        located_fp = pred[pred["located_matched"]].groupby(_GROUP_KEYS).size()

        per_sample = pd.concat([gt_count, pr_count, located_tp], axis=1).fillna(0)
        per_sample["located_pr_count"] = tp_count.reindex(
            per_sample.index, fill_value=0
        ) + located_fp.reindex(per_sample.index, fill_value=0)
        per_sample["located_precision"] = per_sample["located_tp"] / per_sample["pr_count"]
        per_sample["located_recall"] = per_sample["located_tp"] / per_sample["gt_count"]
        # A model whose predictions never touch any real GT cell (precision
        # and recall both 0, e.g. randomly-placed hallucinations) must score
        # located_count_f1 == 0, not NaN from a literal 0/0 - the whole point
        # of this metric is to catch that case as a bad score, not a missing
        # one.
        per_sample["located_count_f1"] = per_sample.apply(
            lambda r: _safe_f1(r["located_precision"], r["located_recall"]), axis=1
        )
        per_sample["abs_pct_error"] = (
            (per_sample["pr_count"] - per_sample["gt_count"]).abs() / per_sample["gt_count"] * 100
        )
        per_sample = per_sample.reset_index()

        # Macro-average(每個樣本先各自算一次再平均)當主要排名依據 - pooled
        # (先加總全部樣本再算一次)只當補充欄位,因為它會讓「樣本 A 講太多、
        # 樣本 B 講太少」這種跨樣本正負誤差互相抵銷,把單一樣本誤差很大的模型
        # 評得比實際情況好。
        summary_rows = []
        for model, g in per_sample.groupby("model"):
            pooled_gt, pooled_pr, pooled_located_tp = (
                g["gt_count"].sum(),
                g["pr_count"].sum(),
                g["located_tp"].sum(),
            )
            pooled_precision = pooled_located_tp / pooled_pr if pooled_pr else float("nan")
            pooled_recall = pooled_located_tp / pooled_gt if pooled_gt else float("nan")
            summary_rows.append(
                {
                    "model": model,
                    "n_samples": len(g),
                    "located_count_f1_macro": g["located_count_f1"].mean(),
                    "mean_abs_pct_error": g["abs_pct_error"].mean(),
                    "pearson_r": (
                        g["gt_count"].corr(g["pr_count"]) if len(g) > 1 else float("nan")
                    ),
                    "located_count_f1_pooled": _safe_f1(pooled_precision, pooled_recall),
                    "mean_abs_pct_error_pooled": (
                        abs(pooled_pr - pooled_gt) / pooled_gt * 100 if pooled_gt else float("nan")
                    ),
                }
            )
        summary = pd.DataFrame(summary_rows).sort_values(
            "located_count_f1_macro", ascending=False, ignore_index=True
        )

        sns.set_theme(style="whitegrid")

        # Scatter: gt_count vs. pr_count (+ located_pr_count), y=x diagonal.
        fig_scatter, ax_scatter = plt.subplots(figsize=(7, 7))
        max_count = max(per_sample["gt_count"].max(), per_sample["pr_count"].max(), 1)
        ax_scatter.plot([0, max_count], [0, max_count], linestyle="--", color="gray", label="y = x")
        sns.scatterplot(
            data=per_sample,
            x="gt_count",
            y="pr_count",
            hue="model",
            style="model",
            s=70,
            ax=ax_scatter,
        )
        sns.scatterplot(
            data=per_sample,
            x="gt_count",
            y="located_pr_count",
            hue="model",
            marker="x",
            legend=False,
            alpha=0.6,
            ax=ax_scatter,
        )
        ax_scatter.set_xlabel("GT count")
        ax_scatter.set_ylabel("Predicted count (o = pr_count, x = located_pr_count)")
        ax_scatter.set_title("GT vs. Predicted Cell Count")
        ax_scatter.legend(loc="best")
        plt.tight_layout()

        # Bland-Altman: x = mean(gt_count, pr_count), y = pr_count - gt_count.
        mean_count = (per_sample["gt_count"] + per_sample["pr_count"]) / 2
        diff_count = per_sample["pr_count"] - per_sample["gt_count"]
        bias = float(diff_count.mean())
        sd = float(diff_count.std())
        loa_upper, loa_lower = bias + 1.96 * sd, bias - 1.96 * sd

        fig_ba, ax_ba = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=mean_count, y=diff_count, hue=per_sample["model"], s=60, ax=ax_ba)
        ax_ba.axhline(bias, color="black", linestyle="-", label=f"mean diff = {bias:.2f}")
        ax_ba.axhline(loa_upper, color="red", linestyle="--", label=f"+1.96 SD = {loa_upper:.2f}")
        ax_ba.axhline(loa_lower, color="red", linestyle="--", label=f"-1.96 SD = {loa_lower:.2f}")
        ax_ba.set_xlabel("Mean of GT count and predicted count")
        ax_ba.set_ylabel("Predicted count - GT count")
        ax_ba.set_title("Bland-Altman: GT vs. Predicted Cell Count")
        ax_ba.legend(loc="best")
        plt.tight_layout()

        bland_altman_table = per_sample[_GROUP_KEYS].copy()
        bland_altman_table["mean_count"] = mean_count
        bland_altman_table["diff_count"] = diff_count

        return [
            ReportArtifact(name="cell_count_agreement_summary", table=summary),
            ReportArtifact(name="cell_count_agreement_per_sample", table=per_sample),
            ReportArtifact(
                name="cell_count_agreement_scatter",
                table=per_sample[_GROUP_KEYS + ["gt_count", "pr_count", "located_pr_count"]],
                figure=fig_scatter,
            ),
            ReportArtifact(
                name="cell_count_agreement_bland_altman",
                table=bland_altman_table,
                figure=fig_ba,
                metadata={
                    "bias": bias,
                    "sd": sd,
                    "loa_upper": loa_upper,
                    "loa_lower": loa_lower,
                },
            ),
        ]
