"""Is the MLP underfitting, or at the data's signal ceiling? Train a histogram
gradient-boosted tree (usual tabular champ) on the same whiff task and compare AUC.
GBM ~= MLP  => at the ceiling (not underfit).   GBM >> MLP => MLP leaves signal."""
import datetime as dt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from whyplus.model.data import FULL_START, load_statcast
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp

WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
y = df["description"].isin(WHIFF).to_numpy(int)
X = df[INPUT_COLS].to_numpy("float32")
rng = np.random.default_rng(0)
va = np.zeros(len(df), bool); va[rng.choice(len(df), int(0.15*len(df)), replace=False)] = True
gbm = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08, l2_regularization=1.0,
                                     early_stopping=True, validation_fraction=0.1, random_state=0)
gbm.fit(X[~va], y[~va])
tr_auc = roc_auc_score(y[~va], gbm.predict_proba(X[~va])[:, 1])
va_auc = roc_auc_score(y[va], gbm.predict_proba(X[va])[:, 1])
print(f"HistGBM  train AUC {tr_auc:.4f}   val AUC {va_auc:.4f}   (MLP val AUC was ~0.759)")
print(f"iterations used: {gbm.n_iter_}")
