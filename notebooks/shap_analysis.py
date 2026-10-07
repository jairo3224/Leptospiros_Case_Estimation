"""Step 3: TreeSHAP explanations for the full XGBoost model.

Usage:  python shap_analysis.py --horizon 2 [--model xgboost|catboost]
Outputs (results/): shap_beeswarm_h*.png, shap_groups_h*.png, shap_importance_h*.csv
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import shap

from common import (FEATURE_GROUPS, FEATURE_SETS, RESULTS_DIR, load_data, make_catboost,
                    make_xgb, split_by_year, target_col)

ap = argparse.ArgumentParser()
ap.add_argument("--horizon", type=int, default=2)
ap.add_argument("--model", choices=["xgboost", "catboost"], default="xgboost")
args = ap.parse_args()
h, which = args.horizon, args.model

df = load_data()
train, test = split_by_year(df, h)
cols = FEATURE_SETS["full"]
model = make_xgb("count:poisson") if which == "xgboost" else make_catboost("Poisson")
model.fit(train[cols], train[target_col(h)])

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(test[cols])  # values are on the log scale for the Poisson objective

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
plt.figure()
shap.summary_plot(sv, test[cols], show=False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"shap_beeswarm_{which}_h{h}.png", dpi=200)
plt.close()

importance = pd.Series(abs(sv).mean(axis=0), index=cols).sort_values(ascending=False)
importance.to_csv(RESULTS_DIR / f"shap_importance_{which}_h{h}.csv", header=["mean_abs_shap"])
print("Top features by mean |SHAP|:")
print(importance.head(10).round(4).to_string())

# Group the SHAP importance by feature category (sum of mean |SHAP|)
groups = {g: importance[[c for c in feats if c in importance.index]].sum()
          for g, feats in FEATURE_GROUPS.items()}
groups = pd.Series(groups).sort_values()
groups.plot(kind="barh", title=f"Grouped SHAP importance ({which}, horizon h={h})")
plt.xlabel("Sum of mean |SHAP| (log-count scale)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"shap_groups_{which}_h{h}.png", dpi=200)
print("\nGrouped importance:")
print(groups.sort_values(ascending=False).round(4).to_string())
