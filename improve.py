"""
Evaluate and improve the Tokyo electricity demand model.

Starting point: the Gradient Boosting model from the prediction project (original_model.py, unchanged).
Steps
  1. Evaluate the original model with several metrics and time-series cross-validation.
  2. Find its weaknesses: errors by hour, day type, special periods, demand level and month.
  3. Improve it step by step:
       Step 1  better preprocessing: holiday-period and bridge-day features, extra temperature features
       Step 2  recent demand history known one day ahead (same hour 2 days ago and 1 week ago)
       Step 3  hyperparameter tuning with time-series cross-validation
       Step 4  different algorithms on the same features: LightGBM, Random Forest, Extra Trees
  4. Compare all models and select the final one by cross-validation (never by the test set).

Run:  python improve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

import original_model as orig

HERE = Path(__file__).resolve().parent
RS = 42
CV = TimeSeriesSplit(n_splits=4, test_size=24 * 120)   # 4 folds, each validates on the next ~4 months

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK_2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#c9c8c2"

# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
CALENDAR_V1 = ["bridge_day", "prev_day_off", "next_day_off", "off_run_length", "holiday_period"]
WEATHER_V1 = ["temp_kanto_min_today_c", "temp_kanto_3h_ago_c", "heating_degrees_24h", "cooling_degrees_24h"]
LAGS_V2 = ["demand_lag48h", "demand_lag168h", "demand_d2_mean", "demand_d2_max"]
FEATURES_V0 = orig.FEATURES
FEATURES_V1 = FEATURES_V0 + CALENDAR_V1 + WEATHER_V1
FEATURES_V2 = FEATURES_V1 + LAGS_V2

LABELS = {**orig.FEATURE_LABELS,
          "bridge_day": "Bridge day (workday between days off)", "prev_day_off": "Previous day off",
          "next_day_off": "Next day off", "off_run_length": "Length of the days-off block",
          "holiday_period": "Holiday period (Golden Week / Obon / New Year)",
          "temp_kanto_min_today_c": "Today's min Kanto temp.", "temp_kanto_3h_ago_c": "Kanto temp. 3 h ago",
          "heating_degrees_24h": "Heating degrees, 24 h mean", "cooling_degrees_24h": "Cooling degrees, 24 h mean",
          "demand_lag48h": "Demand, same hour 2 days ago", "demand_lag168h": "Demand, same hour 1 week ago",
          "demand_d2_mean": "Average demand 2 days ago", "demand_d2_max": "Peak demand 2 days ago"}


def add_improved_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    day = out.index.normalize()

    # --- calendar: what kind of day is it, and what is around it? ---------------------
    daily = pd.Series(out["is_workday"].groupby(day).first().astype(int))
    off = 1 - daily
    prev_off, next_off = off.shift(1, fill_value=0), off.shift(-1, fill_value=0)
    run_id = (off != off.shift()).cumsum()
    run_len = off.groupby(run_id).transform("sum") * off            # e.g. 9 for a 9-day Golden Week
    md = daily.index.month * 100 + daily.index.day
    period = np.select([(md >= 429) & (md <= 506), (md >= 810) & (md <= 818), (md >= 1227) | (md <= 104)],
                       [1, 2, 3], 0)                                 # 1 Golden Week, 2 Obon, 3 New Year
    cal = pd.DataFrame({"bridge_day": ((daily == 1) & (prev_off == 1) & (next_off == 1)).astype(int),
                        "prev_day_off": prev_off, "next_day_off": next_off, "off_run_length": run_len,
                        "holiday_period": period}, index=daily.index)
    for c in CALENDAR_V1:
        out[c] = cal[c].reindex(day).to_numpy()

    # --- weather: cold mornings, heat that has built up, temperature trend ------------
    out["temp_kanto_min_today_c"] = out.groupby(day)["temp_kanto_c"].transform("min")
    out["temp_kanto_3h_ago_c"] = out["temp_kanto_c"].shift(3).bfill()
    out["heating_degrees_24h"] = out["heating_degrees"].rolling(24, min_periods=1).mean()
    out["cooling_degrees_24h"] = out["cooling_degrees"].rolling(24, min_periods=1).mean()

    # --- recent demand that is already known when forecasting the next day ------------
    out["demand_lag48h"] = out["demand_gw"].shift(48)
    out["demand_lag168h"] = out["demand_gw"].shift(168)
    d = out["demand_gw"].groupby(day)
    out["demand_d2_mean"] = d.mean().shift(2).reindex(day).to_numpy()
    out["demand_d2_max"] = d.max().shift(2).reindex(day).to_numpy()
    return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def evaluate(y, p, frame) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    peak = y >= np.quantile(y, 0.95)
    special = (frame["holiday_period"] > 0) | (frame["bridge_day"] == 1)
    ape = np.abs(p - y) / y * 100
    return {"MAE_GW": mean_absolute_error(y, p), "RMSE_GW": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE_pct": ape.mean(), "R2": r2_score(y, p), "max_error_GW": max_error(y, p),
            "bias_GW": float(np.mean(p - y)), "peak_hours_MAPE_pct": ape[peak].mean(),
            "special_days_MAPE_pct": ape[special.to_numpy()].mean(),
            "share_within_3pct": (ape <= 3).mean() * 100}


def cv_mape(model, X, y) -> tuple[float, float]:
    scores = []
    for tr, va in CV.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        scores.append(mean_absolute_percentage_error(y.iloc[va], m.predict(X.iloc[va])) * 100)
    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def original_gb():
    return orig.build_models()["Gradient Boosting"]


def tune_gb(X, y):
    space = {"learning_rate": [0.03, 0.05, 0.08], "max_leaf_nodes": [31, 48, 63, 95],
             "min_samples_leaf": [10, 20, 40, 80], "l2_regularization": [0.0, 0.5, 1.0, 3.0],
             "max_iter": [600, 1000, 1500], "max_features": [0.5, 0.8, 1.0]}
    search = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=RS), space, n_iter=20, cv=CV,
                                scoring="neg_mean_absolute_percentage_error", random_state=RS, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def alternatives():
    return {
        "LightGBM": lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.03, num_leaves=63, min_child_samples=20,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                      random_state=RS, verbose=-1, n_jobs=-1),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, max_features=0.5, n_jobs=-1,
                                               random_state=RS),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.7, n_jobs=-1,
                                           random_state=RS),
    }


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def _style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _title(ax, title, sub, pad=24):
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=pad)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, color=INK_2, va="bottom")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(HERE / name, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def chart_weaknesses(w: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1]})
    a = axes[0]
    _style(a)
    hr = w["by_hour"]
    a.bar(hr.index, hr.values, color=BLUE, width=0.75)
    a.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    a.set_ylabel("Mean absolute % error", color=INK_2, fontsize=9)
    _title(a, "Errors by hour of day", "Original model, test set 2026")
    b = axes[1]
    _style(b, grid="x")
    dt = w["by_day_type"].sort_values()
    colors = [ORANGE if v == dt.max() else BLUE for v in dt.values]
    b.barh(dt.index, dt.values, color=colors, height=0.6)
    for i, v in enumerate(dt.values):
        b.text(v + 0.05, i, f"{v:.1f} %", va="center", fontsize=8.5, color=INK)
    b.set_xlabel("Mean absolute % error", color=INK_2, fontsize=9)
    _title(b, "Errors by type of day", "Bridge days and public holidays are the weak spots")
    _save(fig, "fig1_original_model_weaknesses.png")


def chart_residuals(test, p0):
    res = p0 - test["demand_gw"]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    _style(ax, grid="both")
    ax.scatter(test["demand_gw"], res, s=5, alpha=0.3, color=BLUE, edgecolors="none")
    ax.axhline(0, color=INK_2, linewidth=1)
    bins = pd.cut(test["demand_gw"], np.arange(18, 60, 3))
    m = res.groupby(bins, observed=True).mean()
    ax.plot([b.mid for b in m.index], m.values, color=ORANGE, linewidth=2.2, marker="o", markersize=5,
            label="Average error at that demand level")
    ax.set_xlabel("Actual demand (GW)", color=INK_2, fontsize=9)
    ax.set_ylabel("Prediction − actual (GW)", color=INK_2, fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    _title(ax, "Residuals of the original model",
           "Above 0 = predicted too high. It over-predicts by about 0.4 GW at almost every level: "
           "2026 demand ran ~4 % below 2024–25.")
    _save(fig, "fig2_original_residuals.png")


def chart_steps(comp: pd.DataFrame, final: str):
    c = comp.copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=SURFACE)
    _style(ax, grid="x")
    y = np.arange(len(c))
    ax.barh(y - 0.2, c["cv_MAPE_pct"], height=0.38, color=NEUTRAL, label="Cross-validation (training years)")
    ax.barh(y + 0.2, c["MAPE_pct"], height=0.38,
            color=[ORANGE if n == final else BLUE for n in c.index], label="Test set 2026")
    for i, (cv_, t) in enumerate(zip(c["cv_MAPE_pct"], c["MAPE_pct"])):
        ax.text(t + 0.04, i + 0.2, f"{t:.2f} %", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, c.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute percentage error (lower is better)", color=INK_2, fontsize=9)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.11), frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    _title(ax, f"Error fell from {c['MAPE_pct'].iloc[0]:.2f} % to {c.loc[final, 'MAPE_pct']:.2f} %",
           "Each step adds to the one before. Orange: final model, chosen by cross-validation.")
    _save(fig, "fig3_improvement_steps.png")


def chart_week(test, p0, pf, final, start="2026-09-16", days=11, name="fig4_silver_week_before_after.png",
               label="Silver Week 2026 (5 days off, 19–23 Sep)"):
    sl = slice(start, (pd.Timestamp(start) + pd.Timedelta(days=days)).strftime("%Y-%m-%d"))
    a = test.loc[sl, "demand_gw"]
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    _style(ax)
    ax.plot(a.index, a, color=INK, linewidth=2, label="Actual")
    ax.plot(a.index, p0.loc[sl], color=BLUE, linewidth=1.6, linestyle="--",
            label=f"Original ({mean_absolute_percentage_error(a, p0.loc[sl]) * 100:.1f} % error)")
    ax.plot(a.index, pf.loc[sl], color=ORANGE, linewidth=1.8,
            label=f"Final: {final} ({mean_absolute_percentage_error(a, pf.loc[sl]) * 100:.1f} % error)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d %b"))
    ax.set_ylabel("GW", color=INK_2, fontsize=9)
    ax.set_ylim(top=a.max() * 1.18)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=3)
    _title(ax, f"Before vs after: {label}", "The original model saw each day off on its own; the final "
           "model knows it is inside a long holiday block")
    _save(fig, name)


def chart_month(test, p0, pf, final):
    y = test["demand_gw"]
    m0 = ((p0 - y).abs() / y).groupby(test.index.month).mean() * 100
    mf = ((pf - y).abs() / y).groupby(test.index.month).mean() * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(len(m0))
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    _style(ax)
    ax.bar(x - 0.2, m0.values, width=0.38, color=BLUE, label="Original model")
    ax.bar(x + 0.2, mf.values, width=0.38, color=ORANGE, label=f"Final model ({final})")
    ax.set_xticks(x, [months[i - 1] for i in m0.index])
    ax.set_ylabel("Mean absolute % error", color=INK_2, fontsize=9)
    ax.set_ylim(0, max(m0.max(), mf.max()) * 1.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK, ncol=2)
    better = int((mf.values < m0.values).sum())
    _title(ax, f"Error by month: lower in {better} of {len(m0)} months",
           "Test set, Jan – Oct 2026. August (Obon holidays) is still the hardest month.")
    _save(fig, "fig5_error_by_month_before_after.png")


def chart_importance(imp: pd.Series, final: str):
    imp = imp.head(15).sort_values()
    new = set(CALENDAR_V1 + WEATHER_V1 + LAGS_V2)
    fig, ax = plt.subplots(figsize=(9.5, 5.6), facecolor=SURFACE)
    _style(ax, grid="x")
    ax.barh([LABELS[f] for f in imp.index], imp.values,
            color=[ORANGE if f in new else BLUE for f in imp.index], height=0.65)
    ax.set_xlabel("Increase in test MAE (GW) when the feature is shuffled", color=INK_2, fontsize=9)
    _title(ax, "Top 15 features of the final model", f"{final}, permutation importance. Orange: new features "
           "added in this project")
    _save(fig, "fig6_final_feature_importance.png")


# --------------------------------------------------------------------------------------
def run(save: bool = True) -> dict:
    df = add_improved_features(orig.add_features(orig.load()))
    train = df[df.index < orig.TEST_START]
    test = df[df.index >= orig.TEST_START]
    y_tr, y_te = train["demand_gw"], test["demand_gw"]

    results, preds, models = {}, {}, {}

    def record(name, model, feats, fitted=None):
        cv_m, cv_s = cv_mape(model, train[feats], y_tr)
        m = fitted if fitted is not None else clone(model).fit(train[feats], y_tr)
        p = pd.Series(m.predict(test[feats]), index=test.index)
        results[name] = {"features": len(feats), "cv_MAPE_pct": cv_m, "cv_std": cv_s, **evaluate(y_te, p, test)}
        preds[name], models[name] = p, (m, feats)

    # Original + steps 1 and 2 (same algorithm and settings, only the features change)
    record("Original model (Task 06)", original_gb(), FEATURES_V0)
    record("Step 1: + calendar & weather features", original_gb(), FEATURES_V1)
    record("Step 2: + recent demand history", original_gb(), FEATURES_V2)
    # Step 3: tuning
    tuned, best_params = tune_gb(train[FEATURES_V2], y_tr)
    record("Step 3: + tuned Gradient Boosting", tuned, FEATURES_V2,
           fitted=clone(tuned).fit(train[FEATURES_V2], y_tr))
    # Step 4: other algorithms on the improved features
    for name, model in alternatives().items():
        record(f"Step 4: {name}", model, FEATURES_V2)

    comp = pd.DataFrame(results).T
    final = comp.iloc[1:]["cv_MAPE_pct"].astype(float).idxmin()       # chosen by cross-validation only
    final_model, final_feats = models[final]
    p0, pf = preds["Original model (Task 06)"], preds[final]

    # Weakness analysis of the original model
    ape0 = (p0 - y_te).abs() / y_te * 100
    day_type = np.select([test["bridge_day"] == 1, test["holiday_period"] > 0, test["is_holiday"] == 1,
                          test["is_workday"] == 0], ["Bridge day", "Holiday period (GW/Obon/New Year)",
                                                     "Other public holiday", "Weekend"], "Normal workday")
    weak = {"by_hour": ape0.groupby(test["hour"]).mean(),
            "by_day_type": ape0.groupby(day_type).mean(),
            "hours_by_day_type": pd.Series(day_type).value_counts(),
            "by_demand_level": (p0 - y_te).groupby(pd.qcut(y_te, 5)).mean(),
            "worst_days": ape0.groupby(test.index.normalize()).mean().nlargest(8)}
    weak_after = (pf - y_te).abs().div(y_te).mul(100).groupby(day_type).mean()

    imp = permutation_importance(final_model, test[final_feats], y_te, scoring="neg_mean_absolute_error",
                                 n_repeats=3, random_state=RS, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=final_feats).sort_values(ascending=False)

    out = {"df": df, "train": train, "test": test, "comparison": comp, "final": final, "final_model": final_model,
           "final_features": final_feats, "best_params": best_params, "preds": preds, "weak": weak,
           "weak_after": weak_after, "importance": importance}
    if save:
        comp.round(3).to_csv(HERE / "model_comparison.csv", index_label="model")
        pd.DataFrame({"original": weak["by_day_type"], "final": weak_after,
                      "hours": weak["hours_by_day_type"]}).round(2).to_csv(HERE / "error_by_day_type.csv",
                                                                         index_label="day_type")
        pd.DataFrame({"actual_gw": y_te, "original_pred_gw": p0.round(2), "final_pred_gw": pf.round(2)}).to_csv(
            HERE / "test_predictions_before_after.csv", index_label="datetime")
        joblib.dump({"model": final_model, "features": final_feats, "name": final}, HERE / "final_model.joblib",
                    compress=3)
        (HERE / "final_model_info.json").write_text(json.dumps({
            "final_model": final, "selected_by": "lowest time-series cross-validation MAPE on 2022-2025",
            "tuned_gradient_boosting_params": {k: (float(v) if isinstance(v, (np.floating,)) else v)
                                               for k, v in best_params.items()},
            "features": final_feats, "test_metrics": {k: round(float(v), 4) for k, v in comp.loc[final].items()},
            "original_test_metrics": {k: round(float(v), 4) for k, v in comp.iloc[0].items()}}, indent=2))
        chart_weaknesses(weak)
        chart_residuals(test, p0)
        chart_steps(comp, final)
        chart_week(test, p0, pf, final)
        chart_month(test, p0, pf, final)
        chart_importance(importance, final)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    r = run()
    cols = ["cv_MAPE_pct", "MAE_GW", "RMSE_GW", "MAPE_pct", "R2", "max_error_GW", "bias_GW",
            "peak_hours_MAPE_pct", "special_days_MAPE_pct", "share_within_3pct"]
    print(r["comparison"][cols].astype(float).round(3))
    print("\nFinal model:", r["final"])
    print("Tuned params:", r["best_params"])
    print("\nOriginal errors by day type:\n", r["weak"]["by_day_type"].round(2), "\n", r["weak"]["hours_by_day_type"])
    print("\nAfter:\n", r["weak_after"].round(2))
    print("\nBias by demand level (original):\n", r["weak"]["by_demand_level"].round(2))
    print("\nWorst days (original):\n", r["weak"]["worst_days"].round(2))
    print("\nTop features:\n", r["importance"].head(10).round(3))
