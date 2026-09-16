"""
Malaysia Tourism Pressure — internal model inspection tool
DOSM Datathon 2026

PURPOSE
    Internal validation and CSV export for the Power BI dashboard.
    This app is NOT the submission deliverable. The deliverable is the .pbix
    file, which must open offline with no external dependencies (D-15).

RUN
    pip install -r requirements.txt
    streamlit run app.py

REQUIRED FILES (same folder as app.py)
    master_panel_state_year.csv
    population_state.csv

EVIDENCE TIERING
    Everything this app produces is Tier 3 (model output) and is UNCONFIRMED
    until the project lead runs it, reports it in chat, and records it in
    05_RESULTS_LEDGER.md. Nothing here is a benchmark or a target.
"""

import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------------------------------------------------------- constants
SEED = 42
np.random.seed(SEED)

PANEL_FILE = "master_panel_state_year.csv"
POPULATION_FILE = "population_state.csv"

MICRO_TERRITORIES = ["W.P. Putrajaya", "W.P. Labuan"]          # D-10
CLUSTER_FEATURES = ["intensity", "occupancy_pct", "growth_vs_2019"]
MODEL_FEATURES = [
    "previous_growth",
    "lag_growth_vs_2019",
    "lag_intensity",
    "lag_occupancy",
]
TARGET = "log_growth"

# Kaufman & Rousseeuw (1990): >0.5 evidence for clustering, >0.7 strong.
SILHOUETTE_EVIDENCE_THRESHOLD = 0.5
# Dolnicar & Leisch (2010): >0.8 stable, 0.6-0.8 moderate, <0.6 unstable.
STABILITY_STABLE = 0.8
STABILITY_MODERATE = 0.6

st.set_page_config(page_title="Tourism pressure — model inspection", layout="wide")


# ------------------------------------------------------------------- data
@st.cache_data
def load_panel():
    """Master state-year panel. Indicators recomputed, not trusted as stored."""
    df = pd.read_csv(PANEL_FILE)

    df["intensity"] = df["visitors_thousand"] / df["population_thousand"]

    base = df.loc[df["year"] == 2019, ["state", "visitors_thousand"]].rename(
        columns={"visitors_thousand": "visitors_2019"}
    )
    df = df.merge(base, on="state", how="left")
    df["growth_vs_2019"] = df["visitors_thousand"] / df["visitors_2019"]
    df = df.drop(columns=["visitors_2019"])

    df["is_micro_territory"] = df["state"].isin(MICRO_TERRITORIES)
    return df.sort_values(["state", "year"]).reset_index(drop=True)


@st.cache_data
def load_population():
    """Total population by state-year from the OpenDOSM long-format file."""
    pop = pd.read_csv(POPULATION_FILE)
    pop["date"] = pd.to_datetime(pop["date"])
    pop["year"] = pop["date"].dt.year

    total = pop[
        (pop["sex"] == "both")
        & (pop["age"] == "overall")
        & (pop["ethnicity"] == "overall")
    ]
    return total[["state", "year", "population"]].reset_index(drop=True)


def build_features(df):
    """Target (log growth) and one-year lagged predictors, per state. D-11."""
    out = df.sort_values(["state", "year"]).copy()
    out["log_growth"] = out.groupby("state")["visitors_thousand"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    out["previous_growth"] = out.groupby("state")["log_growth"].shift(1)
    out["lag_growth_vs_2019"] = out.groupby("state")["growth_vs_2019"].shift(1)
    out["lag_intensity"] = out.groupby("state")["intensity"].shift(1)
    out["lag_occupancy"] = out.groupby("state")["occupancy_pct"].shift(1)
    return out


# -------------------------------------------------- Solution 1: clustering
@st.cache_data
def run_clustering(df, start_year, k_choice, n_bootstrap):
    window = df[df["year"] >= start_year].copy()

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(window[CLUSTER_FEATURES])

    rows = []
    for k in range(2, 6):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=20)
        labels = model.fit_predict(x_scaled)
        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette_score(x_scaled, labels),
                "inertia": model.inertia_,
            }
        )
    selection = pd.DataFrame(rows)

    # ---- bootstrap stability, resampled at STATE level ----
    # Resampling rows would break the panel structure: each state contributes
    # several correlated year-observations. Resampling states keeps it intact.
    rng = np.random.default_rng(SEED)
    states = window["state"].unique()
    stability_rows = []
    for k in range(2, 6):
        reference = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(
            x_scaled
        )
        aris = []
        for _ in range(n_bootstrap):
            sample = rng.choice(states, size=len(states), replace=True)
            boot = pd.concat([window[window["state"] == s] for s in sample])
            if boot["state"].nunique() < k + 1:
                continue
            boot_scaler = StandardScaler().fit(boot[CLUSTER_FEATURES])
            model = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit(
                boot_scaler.transform(boot[CLUSTER_FEATURES])
            )
            projected = model.predict(boot_scaler.transform(window[CLUSTER_FEATURES]))
            aris.append(adjusted_rand_score(reference, projected))
        arr = np.array(aris)
        stability_rows.append(
            {
                "k": k,
                "mean_ARI": arr.mean(),
                "sd_ARI": arr.std(),
                "pct_above_0.6": (arr > 0.6).mean() * 100,
                "pct_above_0.8": (arr > 0.8).mean() * 100,
                "n_resamples": len(arr),
            }
        )
    stability = pd.DataFrame(stability_rows)

    selection = selection.merge(stability, on="k")

    if k_choice == "auto (highest silhouette)":
        best_k = int(selection.loc[selection["silhouette_score"].idxmax(), "k"])
    else:
        best_k = int(k_choice)

    kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=20)
    window["cluster"] = kmeans.fit_predict(x_scaled)

    ward = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    window["ward_cluster"] = ward.fit_predict(x_scaled)
    ward_ari = adjusted_rand_score(window["cluster"], window["ward_cluster"])

    # Robustness: refit without micro territories, re-standardised so their
    # extreme intensity no longer defines the scale. D-10.
    robust = window[~window["is_micro_territory"]].copy()
    x_robust = StandardScaler().fit_transform(robust[CLUSTER_FEATURES])
    robust["robust_cluster"] = KMeans(
        n_clusters=best_k, random_state=SEED, n_init=20
    ).fit_predict(x_robust)
    robust_ari = adjusted_rand_score(robust["cluster"], robust["robust_cluster"])

    # Profiles in raw units and as z-scores against the window average.
    profiles = window.groupby("cluster")[CLUSTER_FEATURES].mean()
    zprofiles = (profiles - window[CLUSTER_FEATURES].mean()) / window[
        CLUSTER_FEATURES
    ].std()
    zprofiles.columns = [f"z_{c}" for c in zprofiles.columns]
    profiles = profiles.join(zprofiles)
    profiles["n_observations"] = window.groupby("cluster").size()

    ranges = (
        window.groupby("cluster")[CLUSTER_FEATURES]
        .agg(["min", "median", "max"])
        .round(2)
    )

    transitions = window.pivot(index="state", columns="year", values="cluster")
    transitions.columns = [f"cluster_{c}" for c in transitions.columns]
    transitions = transitions.reset_index()
    transitions["changed_cluster"] = transitions.iloc[:, 1:].nunique(axis=1) > 1

    # Readable surrogate for the distance rule. Fitted TO the labels on the
    # same data, so its accuracy is fit, not generalisation.
    tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(
        window[CLUSTER_FEATURES], window["cluster"]
    )
    tree_rules = export_text(tree, feature_names=CLUSTER_FEATURES, decimals=2)
    tree_fidelity = (tree.predict(window[CLUSTER_FEATURES]) == window["cluster"]).mean()

    centroids = pd.DataFrame(
        kmeans.cluster_centers_, columns=[f"z_{c}" for c in CLUSTER_FEATURES]
    )
    centroids.insert(0, "cluster", centroids.index)
    scaling = pd.DataFrame(
        {"feature": CLUSTER_FEATURES, "mean": scaler.mean_, "scale": scaler.scale_}
    )

    return {
        "selection": selection,
        "best_k": best_k,
        "labelled": window,
        "profiles": profiles.reset_index(),
        "ranges": ranges,
        "transitions": transitions,
        "ward_ari": ward_ari,
        "robust_ari": robust_ari,
        "robust": robust,
        "tree_rules": tree_rules,
        "tree_fidelity": tree_fidelity,
        "centroids": centroids,
        "scaling": scaling,
    }


# -------------------------------------------------- Solution 2: projection
def _score(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE (%)": mean_absolute_percentage_error(actual, predicted) * 100,
        "Mean bias (%)": ((predicted - actual) / actual).mean() * 100,
    }


def _fit_supervised(x_train, y_train, x_predict):
    """All five supervised models. SVR and the linear models are scaled;
    SVR in particular is scale-sensitive and is meaningless on raw inputs."""
    scaler = StandardScaler()
    xs_train = scaler.fit_transform(x_train)
    xs_predict = scaler.transform(x_predict)

    ridge = Ridge(alpha=1.0).fit(xs_train, y_train)
    enet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=SEED).fit(xs_train, y_train)
    svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale").fit(xs_train, y_train)
    forest = RandomForestRegressor(
        n_estimators=300, max_depth=4, random_state=SEED
    ).fit(x_train, y_train)
    boost = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED
    ).fit(x_train, y_train)

    return {
        "Ridge Regression": ridge.predict(xs_predict),
        "Elastic Net": enet.predict(xs_predict),
        "SVR": svr.predict(xs_predict),
        "Random Forest": forest.predict(x_predict),
        "Gradient Boosting": boost.predict(x_predict),
    }, {"ridge": ridge, "forest": forest}


@st.cache_data
def run_projection(df, population, train_start, test_year, use_regime_dummy):
    featured = build_features(df)

    features = list(MODEL_FEATURES)
    if use_regime_dummy:
        # Keeps the pandemic years in training but lets the model separate
        # them, instead of pooling incompatible regimes. See D-09 rationale.
        featured["covid_regime"] = featured["year"].isin([2020, 2021, 2022]).astype(int)
        features.append("covid_regime")

    model_df = featured.dropna(subset=features + [TARGET]).copy()
    train_df = model_df[
        (model_df["year"] >= train_start) & (model_df["year"] < test_year)
    ].copy()
    test_df = model_df[model_df["year"] == test_year].copy()

    if len(train_df) < 10 or test_df.empty:
        return {"error": f"Only {len(train_df)} training rows. Widen the window."}

    national = featured.groupby("year")["visitors_thousand"].sum().sort_index()
    national_log_growth = np.log(national / national.shift(1))

    predictions = {
        # Zero-growth: predict no change at all. The floor every method must clear.
        "Zero-growth": np.zeros(len(test_df)),
        "Naive": test_df["previous_growth"].values,
        "National Growth": np.repeat(
            national_log_growth.loc[test_year - 1], len(test_df)
        ),
    }
    supervised, fitted = _fit_supervised(
        train_df[features], train_df[TARGET], test_df[features]
    )
    predictions.update(supervised)

    prior = featured.loc[
        featured["year"] == test_year - 1, ["state", "visitors_thousand"]
    ].rename(columns={"visitors_thousand": "prior_visitors"})

    evaluation = test_df[["state", "visitors_thousand", TARGET]].merge(
        prior, on="state", how="left"
    )
    metric_rows = []
    for name, pred in predictions.items():
        column = f"{name} predicted"
        evaluation[column] = evaluation["prior_visitors"] * np.exp(pred)
        metric_rows.append(
            {"Model": name, **_score(evaluation["visitors_thousand"], evaluation[column])}
        )

    metrics = pd.DataFrame(metric_rows).sort_values("MAPE (%)").reset_index(drop=True)
    best_model = metrics.iloc[0]["Model"]

    importance = pd.DataFrame(
        {"feature": features, "importance": fitted["forest"].feature_importances_}
    ).sort_values("importance", ascending=False)
    coefficients = pd.DataFrame(
        {"feature": features, "ridge_coefficient": fitted["ridge"].coef_}
    )

    # ---- project the forecast year, retraining on everything available ----
    latest_year = featured["year"].max()
    latest = featured[featured["year"] == latest_year].copy()
    forward = latest.copy()
    forward["previous_growth"] = latest["log_growth"]
    forward["lag_growth_vs_2019"] = latest["growth_vs_2019"]
    forward["lag_intensity"] = latest["intensity"]
    forward["lag_occupancy"] = latest["occupancy_pct"]
    if use_regime_dummy:
        forward["covid_regime"] = 0

    final_train = model_df[model_df["year"] >= train_start]
    final_supervised, _ = _fit_supervised(
        final_train[features], final_train[TARGET], forward[features]
    )

    if best_model in final_supervised:
        forward_growth = final_supervised[best_model]
    elif best_model == "Naive":
        forward_growth = forward["log_growth"].values
    elif best_model == "Zero-growth":
        forward_growth = np.zeros(len(forward))
    else:
        forward_growth = np.repeat(national_log_growth.loc[latest_year], len(forward))

    target_year = latest_year + 1
    forward["predicted_log_growth"] = forward_growth
    forward["predicted_growth_pct"] = (np.exp(forward_growth) - 1) * 100
    forward["projected_visitors"] = forward["visitors_thousand"] * np.exp(forward_growth)

    pop_target = population[population["year"] == target_year][
        ["state", "population"]
    ].rename(columns={"population": "population_thousand_target"})
    projection = forward.merge(pop_target, on="state", how="left")
    projection["projected_intensity"] = (
        projection["projected_visitors"] / projection["population_thousand_target"]
    )

    projection_output = projection[
        [
            "state",
            "visitors_thousand",
            "intensity",
            "predicted_log_growth",
            "predicted_growth_pct",
            "projected_visitors",
            "population_thousand_target",
            "projected_intensity",
        ]
    ].rename(
        columns={
            "visitors_thousand": f"visitors_{latest_year}_thousand",
            "intensity": f"intensity_{latest_year}",
        }
    )

    return {
        "metrics": metrics,
        "best_model": best_model,
        "validation": evaluation,
        "importance": importance,
        "coefficients": coefficients,
        "projection": projection_output,
        "train_rows": len(train_df),
        "train_years": sorted(train_df["year"].unique().tolist()),
        "test_rows": len(test_df),
        "national_log_growth": national_log_growth,
        "target_year": target_year,
        "latest_year": latest_year,
        "features_used": features,
        "train_target_range": (train_df[TARGET].min(), train_df[TARGET].max()),
        "test_target_range": (test_df[TARGET].min(), test_df[TARGET].max()),
    }


# ----------------------------------------------- Solution 3: early warning
def run_early_warning(df, projection, uplift_pct, latest_year):
    peak = (
        df.groupby("state")["intensity"]
        .max()
        .reset_index()
        .rename(columns={"intensity": "historical_peak_intensity"})
    )
    peak_year = (
        df.loc[df.groupby("state")["intensity"].idxmax()][["state", "year"]]
        .rename(columns={"year": "peak_year"})
    )

    warning = projection.merge(peak, on="state", how="left").merge(
        peak_year, on="state", how="left"
    )
    warning["increase_over_peak"] = (
        warning["projected_intensity"] - warning["historical_peak_intensity"]
    )
    warning["pct_over_peak"] = (
        warning["projected_intensity"] / warning["historical_peak_intensity"] - 1
    ) * 100

    factor = 1 + uplift_pct / 100
    warning["scenario_visitors"] = warning["projected_visitors"] * factor
    warning["scenario_intensity"] = (
        warning["scenario_visitors"] / warning["population_thousand_target"]
    )
    warning["scenario_increase_over_peak"] = (
        warning["scenario_intensity"] - warning["historical_peak_intensity"]
    )

    warning = warning.sort_values("increase_over_peak", ascending=False).reset_index(
        drop=True
    )
    warning["priority_rank"] = warning.index + 1

    # Degeneracy diagnostic: if the ranking simply reproduces current
    # intensity, it carries no information beyond what Solution 1 shows.
    current = f"intensity_{latest_year}"
    rho, pval = spearmanr(warning["increase_over_peak"], warning[current])
    flagged = int((warning["increase_over_peak"] > 0).sum())
    peaks_in_latest = int((warning["peak_year"] == latest_year).sum())

    return warning, {
        "rho": rho,
        "p": pval,
        "flagged": flagged,
        "total": len(warning),
        "peaks_in_latest": peaks_in_latest,
    }


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
st.sidebar.caption(
    "Every control here is a methodology choice. Whatever is selected when you "
    "export is what the CSVs contain."
)

st.sidebar.subheader("Clustering")
cluster_start = st.sidebar.selectbox(
    "Window starts", [2018, 2021, 2022, 2023], index=3,
    help="D-09 restricts clustering to 2023 onward.",
)
k_choice = st.sidebar.selectbox(
    "Clusters (k)", ["auto (highest silhouette)", "2", "3", "4", "5"], index=0
)
n_bootstrap = st.sidebar.slider("Bootstrap resamples", 100, 1000, 500, step=100)

st.sidebar.subheader("Projection")
train_start = st.sidebar.selectbox(
    "Training starts", [2020, 2021, 2022, 2023], index=0,
    help=(
        "2020 pools the pandemic collapse, the 2022 rebound and the settled "
        "years. 2023 trains only on post-recovery years but leaves few rows. "
        "UNRESOLVED — see the Diagnostics tab."
    ),
)
use_regime_dummy = st.sidebar.checkbox(
    "Add COVID regime indicator", value=False,
    help="Keeps the pandemic years but lets the model separate them.",
)

st.sidebar.subheader("Scenario")
uplift_pct = st.sidebar.slider("VMY 2026 uplift (%)", 0, 30, 10,
                               help="D-14: illustrative control, not a model output.")

st.sidebar.divider()
st.sidebar.caption(f"Seed fixed at {SEED}. scikit-learn defaults unless noted.")

# ------------------------------------------------------------------- load
try:
    panel = load_panel()
    population = load_population()
except FileNotFoundError as exc:
    st.error(
        f"Missing file: {exc.filename}. Put {PANEL_FILE} and {POPULATION_FILE} "
        "in the same folder as app.py."
    )
    st.stop()

clustering = run_clustering(panel, cluster_start, k_choice, n_bootstrap)
projection_result = run_projection(panel, population, train_start, 2025, use_regime_dummy)

st.title("Malaysia tourism pressure — model inspection")
st.caption(
    "Internal tool for checking model behaviour and exporting result CSVs. "
    "Not the submission dashboard — that is the .pbix (D-15)."
)
st.warning(
    "Everything on this page is Tier 3 exploratory output. Nothing is a result "
    "until the project lead runs it and records it in 05_RESULTS_LEDGER.md.",
    icon="⚠️",
)

tabs = st.tabs(
    ["Pressure profiles", "Projection", "Early warning", "Diagnostics", "Export"]
)

# ------------------------------------------------------- tab 1: clustering
with tabs[0]:
    selection = clustering["selection"]
    best_k = clustering["best_k"]
    best_row = selection[selection["k"] == best_k].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters", best_k)
    c2.metric("Silhouette", f"{best_row['silhouette_score']:.4f}")
    c3.metric("Bootstrap ARI", f"{best_row['mean_ARI']:.3f}")
    c4.metric("Ward agreement", f"{clustering['ward_ari']:.3f}")

    if best_row["silhouette_score"] < SILHOUETTE_EVIDENCE_THRESHOLD:
        st.warning(
            f"Silhouette {best_row['silhouette_score']:.4f} is below the "
            f"conventional {SILHOUETTE_EVIDENCE_THRESHOLD} threshold for evidence "
            "of clustering (Kaufman & Rousseeuw 1990). Report this, do not omit it.",
            icon="⚠️",
        )
    if best_row["mean_ARI"] < STABILITY_STABLE:
        band = "moderate" if best_row["mean_ARI"] >= STABILITY_MODERATE else "unstable"
        st.warning(
            f"Bootstrap stability {best_row['mean_ARI']:.3f} is {band}, not stable "
            "(Dolnicar & Leisch 2010: >0.8 stable, 0.6-0.8 moderate).",
            icon="⚠️",
        )

    top = selection.nlargest(2, "silhouette_score")
    margin = top["silhouette_score"].iloc[0] - top["silhouette_score"].iloc[1]
    if margin < 0.05:
        st.info(
            f"k={int(top['k'].iloc[0])} and k={int(top['k'].iloc[1])} differ by only "
            f"{margin:.4f} in silhouette. k is not decided by the metric — choose on "
            "interpretability and say so (D-21).",
            icon="ℹ️",
        )

    st.subheader("How many groups the data supports")
    st.dataframe(selection.round(4), width="stretch", hide_index=True)

    left, right = st.columns(2)
    left.markdown("**Silhouette by k**")
    left.line_chart(selection.set_index("k")["silhouette_score"])
    right.markdown("**Bootstrap stability by k**")
    right.line_chart(selection.set_index("k")["mean_ARI"])

    st.subheader("Cluster profiles")
    st.caption(
        "z-scores are against the window average. Name clusters for what the "
        "indicators show, never for an assumed cause (D-17), and as types rather "
        "than severity levels (D-22)."
    )
    st.dataframe(clustering["profiles"].round(3), width="stretch", hide_index=True)

    st.markdown("**Observed ranges per cluster**")
    st.caption("Ranges overlap on every single indicator — no threshold separates them.")
    st.dataframe(clustering["ranges"], width="stretch")

    st.subheader("The assignment rule")
    st.caption(
        "K-means assigns by nearest centroid in standardised space. This is the "
        "exact, complete rule and belongs in the methodology section."
    )
    rc1, rc2 = st.columns(2)
    rc1.markdown("*Standardisation constants*")
    rc1.dataframe(clustering["scaling"].round(4), width="stretch", hide_index=True)
    rc2.markdown("*Centroids (standardised units)*")
    rc2.dataframe(clustering["centroids"].round(4), width="stretch", hide_index=True)
    st.code(
        "z = (value - mean) / scale\n"
        "d_c = sqrt( sum_over_features( (z - centroid_c)^2 ) )\n"
        "cluster = argmin_c d_c",
        language="text",
    )

    with st.expander("Readable approximation of that rule (decision tree surrogate)"):
        st.caption(
            f"Fitted to the labels on the same data — {clustering['tree_fidelity']:.1%} "
            "agreement. This is fit, not accuracy on new data. Present it as a "
            "description, never as the rule that produced the clusters."
        )
        st.code(clustering["tree_rules"], language="text")

    st.subheader("States that move between groups")
    transitions = clustering["transitions"]
    movers = transitions.loc[transitions["changed_cluster"], "state"].tolist()
    st.write(", ".join(movers) if movers else "No state changed group.")
    st.dataframe(transitions, width="stretch", hide_index=True)

    st.subheader("Observations by year")
    year_pick = st.selectbox(
        "Year", sorted(clustering["labelled"]["year"].unique()), key="cluster_year"
    )
    st.dataframe(
        clustering["labelled"]
        .loc[
            clustering["labelled"]["year"] == year_pick,
            ["state"] + CLUSTER_FEATURES + ["cluster", "ward_cluster"],
        ]
        .round(3),
        width="stretch",
        hide_index=True,
    )

# ------------------------------------------------------- tab 2: projection
with tabs[1]:
    if "error" in projection_result:
        st.error(projection_result["error"])
    else:
        metrics = projection_result["metrics"]
        best = metrics.iloc[0]
        zero_row = metrics[metrics["Model"] == "Zero-growth"].iloc[0]

        st.subheader("2025 hold-out validation")
        st.caption(
            f"Trained on {projection_result['train_rows']} rows "
            f"({projection_result['train_years'][0]}–{projection_result['train_years'][-1]}), "
            f"tested on {projection_result['test_rows']} rows from 2025. "
            f"Features: {', '.join(projection_result['features_used'])}."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Lowest MAPE", best["Model"])
        m2.metric("Its MAPE", f"{best['MAPE (%)']:.2f}%")
        m3.metric(
            "vs no model at all",
            f"{best['MAPE (%)'] - zero_row['MAPE (%)']:+.2f} pts",
            help="Zero-growth predicts 2025 = 2024. Anything that cannot beat it "
                 "is adding nothing.",
        )

        if best["Model"] in {"Zero-growth", "Naive", "National Growth"}:
            st.info(
                "A baseline beat every supervised model. Under D-12 this is a "
                "reportable finding and the simpler method is the one to use. "
                "It is not a failure to correct.",
                icon="ℹ️",
            )

        st.dataframe(metrics.round(4), width="stretch", hide_index=True)
        st.caption(
            "Mean bias equal in magnitude to MAPE means the model misses in one "
            "direction on every state — a sign of regime mismatch, not noise."
        )

        st.subheader("Predicted against actual, by state")
        st.dataframe(
            projection_result["validation"]
            .drop(columns=[TARGET])
            .round(1),
            width="stretch",
            hide_index=True,
        )

        st.subheader("What the random forest leaned on")
        st.caption("Describes this fitted model only. Not evidence of causation.")
        st.bar_chart(projection_result["importance"].set_index("feature")["importance"])

        st.subheader(f"{projection_result['target_year']} projection")
        st.caption(
            f"Produced by the lowest-MAPE method ({projection_result['best_model']}), "
            "retrained on all available years. Trend continuation only — contains no "
            "VMY 2026 campaign effect (D-14)."
        )
        display = projection_result["projection"].sort_values(
            "projected_intensity", ascending=False
        )
        st.dataframe(display.round(3), width="stretch", hide_index=True)
        st.bar_chart(display.set_index("state")["projected_intensity"])

# ---------------------------------------------------- tab 3: early warning
with tabs[2]:
    if "error" in projection_result:
        st.error("Projection failed, so the ranking cannot be built.")
    else:
        warning, diag = run_early_warning(
            panel,
            projection_result["projection"],
            uplift_pct,
            projection_result["latest_year"],
        )

        st.subheader("Degeneracy check — read this before using the ranking")
        d1, d2, d3 = st.columns(3)
        d1.metric("Rank corr. with current intensity", f"{diag['rho']:.4f}")
        d2.metric("States flagged", f"{diag['flagged']} of {diag['total']}")
        d3.metric(
            f"Peaks in {projection_result['latest_year']}",
            f"{diag['peaks_in_latest']} of {diag['total']}",
        )

        if abs(diag["rho"]) > 0.9 or diag["flagged"] == diag["total"]:
            st.error(
                f"The ranking reproduces current intensity (Spearman "
                f"{diag['rho']:.4f}) and flags {diag['flagged']} of {diag['total']} "
                "states. Because most states peaked in the latest year and the "
                "winning method applies one growth rate to all of them, a constant "
                "multiplier cannot reorder anything. D-13 anticipated this for a "
                "binary flag; the ranking inherits it. Disclose this explicitly or "
                "present the two dimensions separately.",
                icon="🚨",
            )

        st.subheader("Projected intensity against each state's own peak")
        st.dataframe(
            warning[
                ["priority_rank", "state", "peak_year", "historical_peak_intensity",
                 "projected_intensity", "increase_over_peak", "pct_over_peak"]
            ].round(3),
            width="stretch",
            hide_index=True,
        )

        st.subheader(f"With a {uplift_pct}% uplift applied")
        st.caption(
            "The uplift is an assumption set by the slider, not a model output "
            "(D-14). Applied uniformly, it cannot change the ordering."
        )
        scenario = warning.sort_values(
            "scenario_increase_over_peak", ascending=False
        ).reset_index(drop=True)
        scenario["scenario_rank"] = scenario.index + 1
        st.dataframe(
            scenario[
                ["scenario_rank", "state", "historical_peak_intensity",
                 "scenario_intensity", "scenario_increase_over_peak"]
            ].round(3),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Two-dimensional view")
        st.caption(
            "Projected intensity against 2025 hotel occupancy. Avoids collapsing "
            "two different kinds of pressure into one rank."
        )
        occupancy = panel[panel["year"] == projection_result["latest_year"]][
            ["state", "occupancy_pct"]
        ]
        twod = warning.merge(occupancy, on="state", how="left")
        st.scatter_chart(
            twod, x="occupancy_pct", y="projected_intensity", color="state"
        )

# ------------------------------------------------------ tab 4: diagnostics
with tabs[3]:
    st.subheader("Training window composition")
    st.caption(
        "National year-on-year log growth. Pooling the collapse, the rebound and "
        "the settled years teaches the regressions a relationship that does not "
        "hold in a normal year. D-09 restricted clustering for this reason; the "
        "regression has not inherited that restriction."
    )
    national = projection_result.get("national_log_growth")
    if national is not None:
        st.dataframe(
            national.reset_index()
            .rename(columns={"visitors_thousand": "national_log_growth"})
            .round(4),
            width="stretch",
            hide_index=True,
        )

    if "train_target_range" in projection_result:
        lo, hi = projection_result["train_target_range"]
        tlo, thi = projection_result["test_target_range"]
        st.markdown("**Range check — why tree models cannot escape their training set**")
        st.caption(
            "Random forest and gradient boosting predictions are averages of "
            "training leaves, so they cannot go outside the training target range."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {"set": "training target", "min": lo, "max": hi},
                    {"set": "test actual", "min": tlo, "max": thi},
                ]
            ).round(3),
            width="stretch",
            hide_index=True,
        )

    if "coefficients" in projection_result:
        st.subheader("Fitted ridge coefficients")
        st.caption(
            "A large negative coefficient on lagged recovery means the model "
            "learned that a high prior level predicts a fall — an artefact of the "
            "pandemic years if those are in the training window."
        )
        st.dataframe(
            projection_result["coefficients"].round(4), width="stretch", hide_index=True
        )

    st.subheader("Is there any state-level growth signal to learn?")
    st.caption(
        "Correlation between a state's previous-year growth and its current "
        "growth, across states. If this is not distinguishable from zero, no "
        "model of any family can extract a state-specific signal."
    )
    feat = build_features(panel).dropna(subset=["log_growth"])
    feat["lag"] = feat.groupby("state")["log_growth"].shift(1)
    rows = []
    for yr in sorted(feat["year"].unique()):
        sub = feat[feat["year"] == yr].dropna(subset=["log_growth", "lag"])
        if len(sub) < 5:
            continue
        r, p = spearmanr(sub["lag"], sub["log_growth"])
        rows.append(
            {
                "year": yr,
                "rank_corr": r,
                "p_value": p,
                "cross_state_sd": sub["log_growth"].std(),
            }
        )
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)

    st.subheader("Panel integrity")
    st.dataframe(
        pd.DataFrame(
            [
                {"check": "Rows", "value": str(len(panel))},
                {"check": "States", "value": str(panel["state"].nunique())},
                {"check": "Years", "value": f"{panel['year'].min()}–{panel['year'].max()}"},
                {"check": "Missing values", "value": str(int(panel.isna().sum().sum()))},
                {
                    "check": "State-years absent",
                    "value": str(
                        panel["state"].nunique() * panel["year"].nunique() - len(panel)
                    ),
                },
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Indicator correlations in the clustering window")
    st.dataframe(
        panel[panel["year"] >= cluster_start][CLUSTER_FEATURES]
        .corr()
        .round(4)
        .reset_index(),
        width="stretch",
        hide_index=True,
    )

# ----------------------------------------------------------- tab 5: export
with tabs[4]:
    st.subheader("Result CSVs for Power BI")
    st.caption(
        "Export once the settings are final. Regenerating after the dashboard is "
        "built is how report numbers stop matching dashboard numbers."
    )

    exports = {
        "solution1_k_selection.csv": clustering["selection"],
        "solution1_cluster_results.csv": clustering["labelled"],
        "solution1_cluster_profiles.csv": clustering["profiles"],
        "solution1_cluster_transitions.csv": clustering["transitions"],
        "solution1_robustness_check.csv": clustering["robust"],
        "solution1_centroids.csv": clustering["centroids"],
        "solution1_scaling_constants.csv": clustering["scaling"],
    }

    if "error" not in projection_result:
        warning, diag = run_early_warning(
            panel,
            projection_result["projection"],
            uplift_pct,
            projection_result["latest_year"],
        )
        exports.update(
            {
                "solution2_model_metrics.csv": projection_result["metrics"],
                "solution2_validation.csv": projection_result["validation"],
                "solution2_feature_importance.csv": projection_result["importance"],
                "solution2_ridge_coefficients.csv": projection_result["coefficients"],
                "solution2_projection.csv": projection_result["projection"],
                "solution3_early_warning.csv": warning,
                "solution3_degeneracy_check.csv": pd.DataFrame([diag]),
            }
        )

    exports["run_settings.csv"] = pd.DataFrame(
        [
            {"setting": "seed", "value": SEED},
            {"setting": "clustering_window_start", "value": cluster_start},
            {"setting": "clustering_k", "value": clustering["best_k"]},
            {"setting": "bootstrap_resamples", "value": n_bootstrap},
            {"setting": "regression_train_start", "value": train_start},
            {"setting": "covid_regime_dummy", "value": use_regime_dummy},
            {"setting": "regression_test_year", "value": 2025},
            {"setting": "uplift_pct", "value": uplift_pct},
        ]
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, frame in exports.items():
            archive.writestr(filename, frame.to_csv(index=False))

    st.download_button(
        "Download all result CSVs",
        buffer.getvalue(),
        "tourism_pressure_results.zip",
        "application/zip",
    )

    st.dataframe(
        pd.DataFrame([{"file": n, "rows": len(f)} for n, f in exports.items()]),
        width="stretch",
        hide_index=True,
    )

    st.info(
        "run_settings.csv records the configuration that produced this export. "
        "Keep it with the CSVs — it is what makes the run reproducible.",
        icon="ℹ️",
    )
