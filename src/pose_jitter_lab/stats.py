"""Statistical tests and agreement analysis for cross-domain jitter comparison.

All functions are designed for thesis-grade reporting: they return structured
results with effect sizes, confidence intervals, and interpretive labels that
can be directly embedded in a thesis methods/results section.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Bland-Altman analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BlandAltmanResult:
    """Bland-Altman method-comparison result for one trial+joint pair."""
    trial: str
    joint: str
    n: int
    mean_diff: float
    std_diff: float
    loa_lower: float
    loa_upper: float
    ci_mean_diff: tuple[float, float]
    ci_loa_lower: tuple[float, float]
    ci_loa_upper: tuple[float, float]
    percent_within_loa: float
    means: np.ndarray
    diffs: np.ndarray

    @property
    def bias(self) -> float:
        return self.mean_diff

    @property
    def loa_range(self) -> float:
        return self.loa_upper - self.loa_lower

    def interpret(self) -> str:
        if abs(self.mean_diff) < 0.01 and self.percent_within_loa >= 95.0:
            return "Excellent agreement: negligible bias, ≥95% within LoA."
        if abs(self.mean_diff) < 0.05 and self.percent_within_loa >= 90.0:
            return "Good agreement: small bias, ≥90% within LoA."
        if self.percent_within_loa >= 80.0:
            return "Moderate agreement: some systematic or random differences."
        return "Poor agreement: substantial bias or wide limits of agreement."


def bland_altman(
    values_a: np.ndarray,
    values_b: np.ndarray,
    trial: str = "",
    joint: str = "",
    confidence: float = 0.95,
) -> BlandAltmanResult:
    """Compute Bland-Altman agreement between two matched measurement vectors.

    Parameters
    ----------
    values_a, values_b : array-like
        Paired measurements from system A and system B.
    trial, joint : str
        Labels for identification.
    confidence : float
        Confidence level for CIs on bias and LoA.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        nan_ci = (np.nan, np.nan)
        return BlandAltmanResult(
            trial=trial, joint=joint, n=n,
            mean_diff=np.nan, std_diff=np.nan,
            loa_lower=np.nan, loa_upper=np.nan,
            ci_mean_diff=nan_ci, ci_loa_lower=nan_ci, ci_loa_upper=nan_ci,
            percent_within_loa=np.nan,
            means=np.array([]), diffs=np.array([]),
        )

    diffs = a - b
    means = (a + b) / 2.0
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))

    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    t_crit = float(sp_stats.t.ppf((1 + confidence) / 2, n - 1))
    se_mean = std_diff / np.sqrt(n)
    ci_mean_diff = (mean_diff - t_crit * se_mean, mean_diff + t_crit * se_mean)

    # CI for LoA uses the approximation from Bland & Altman (1999)
    se_loa = np.sqrt(3.0 * std_diff ** 2 / n)
    ci_loa_lower = (loa_lower - t_crit * se_loa, loa_lower + t_crit * se_loa)
    ci_loa_upper = (loa_upper - t_crit * se_loa, loa_upper + t_crit * se_loa)

    within = float(np.sum((diffs >= loa_lower) & (diffs <= loa_upper)) / n * 100)

    return BlandAltmanResult(
        trial=trial, joint=joint, n=n,
        mean_diff=mean_diff, std_diff=std_diff,
        loa_lower=loa_lower, loa_upper=loa_upper,
        ci_mean_diff=ci_mean_diff,
        ci_loa_lower=ci_loa_lower, ci_loa_upper=ci_loa_upper,
        percent_within_loa=within,
        means=means, diffs=diffs,
    )


def bland_altman_table(summary: pd.DataFrame) -> tuple[list[BlandAltmanResult], pd.DataFrame]:
    """Build Bland-Altman results for every matched trial+joint across sources.

    Expects the jitter *summary* DataFrame with columns:
    source, domain, trial, joint, normalized_mean.

    Returns a list of BlandAltmanResult and a summary DataFrame.
    """
    if summary.empty:
        return [], pd.DataFrame()

    pivot = summary.pivot_table(
        index=["trial", "joint"],
        columns="source",
        values="normalized_mean",
        aggfunc="first",
    ).dropna(how="any")

    if pivot.shape[1] < 2:
        return [], pd.DataFrame()

    sources = list(pivot.columns)
    results: list[BlandAltmanResult] = []
    for (trial, joint), row in pivot.iterrows():
        a = np.array([float(row[sources[0]])])
        b = np.array([float(row[sources[1]])])
        results.append(bland_altman(a, b, trial=str(trial), joint=str(joint)))

    table = pd.DataFrame([
        {
            "trial": r.trial,
            "joint": r.joint,
            "n": r.n,
            "bias": r.mean_diff,
            "loa_lower": r.loa_lower,
            "loa_upper": r.loa_upper,
            "pct_within_loa": r.percent_within_loa,
            "interpretation": r.interpret(),
        }
        for r in results
    ])
    return results, table


def bland_altman_from_jitter(jitter: pd.DataFrame) -> tuple[list[BlandAltmanResult], pd.DataFrame]:
    """Build Bland-Altman from frame-level jitter, matching frames across sources.

    This gives per-frame paired comparisons instead of per-summary comparisons,
    which is much more powerful statistically.
    """
    if jitter.empty:
        return [], pd.DataFrame()

    sources = sorted(jitter["source"].unique())
    if len(sources) < 2:
        return [], pd.DataFrame()

    results: list[BlandAltmanResult] = []
    for (trial, joint), group in jitter.groupby(["trial", "joint"]):
        pivot = group.pivot_table(
            index="frame_to",
            columns="source",
            values="normalized_jitter",
            aggfunc="first",
        ).dropna(how="any")
        if pivot.shape[1] < 2 or len(pivot) < 3:
            continue
        a = pivot[sources[0]].values
        b = pivot[sources[1]].values
        results.append(bland_altman(a, b, trial=str(trial), joint=str(joint)))

    if not results:
        return [], pd.DataFrame()

    table = pd.DataFrame([
        {
            "trial": r.trial,
            "joint": r.joint,
            "n": r.n,
            "bias": r.bias,
            "std_diff": r.std_diff,
            "loa_lower": r.loa_lower,
            "loa_upper": r.loa_upper,
            "pct_within_loa": r.percent_within_loa,
            "interpretation": r.interpret(),
        }
        for r in results
    ])
    return results, table


# ---------------------------------------------------------------------------
# Intraclass Correlation Coefficient (ICC)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ICCResult:
    """ICC result with confidence interval and reliability label."""
    icc: float
    ci_lower: float
    ci_upper: float
    f_value: float
    p_value: float
    label: str

    @staticmethod
    def reliability_label(icc: float) -> str:
        if icc < 0.50:
            return "poor"
        if icc < 0.75:
            return "moderate"
        if icc < 0.90:
            return "good"
        return "excellent"


def compute_icc(
    values_a: np.ndarray,
    values_b: np.ndarray,
    confidence: float = 0.95,
) -> ICCResult:
    """Compute ICC(2,1) — two-way random, single measures, absolute agreement.

    This is the appropriate ICC type for comparing two measurement systems
    measuring the same subjects.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = len(a)
    if n < 3:
        return ICCResult(icc=np.nan, ci_lower=np.nan, ci_upper=np.nan,
                         f_value=np.nan, p_value=np.nan, label="insufficient_data")

    k = 2  # two raters/systems
    data = np.column_stack([a, b])
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)

    # Sum of squares
    ss_total = float(np.sum((data - grand_mean) ** 2))
    ss_rows = k * float(np.sum((row_means - grand_mean) ** 2))
    ss_cols = n * float(np.sum((col_means - grand_mean) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-15

    # ICC(2,1)
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)
    icc = float(np.clip(icc, -1.0, 1.0))

    # F-test
    f_value = ms_rows / ms_error if ms_error > 1e-15 else np.inf
    df1, df2 = n - 1, (n - 1) * (k - 1)
    p_value = float(1 - sp_stats.f.cdf(f_value, df1, df2)) if np.isfinite(f_value) else 0.0

    # Confidence interval via F-distribution transformation
    alpha = 1 - confidence
    f_lower = float(sp_stats.f.ppf(alpha / 2, df1, df2))
    f_upper = float(sp_stats.f.ppf(1 - alpha / 2, df1, df2))

    ci_lower = 1 - 1 / (f_lower * f_value) if f_value > 0 else -1.0
    ci_upper = 1 - 1 / (f_upper * f_value) if f_value > 0 else 1.0
    ci_lower = float(np.clip(ci_lower, -1.0, 1.0))
    ci_upper = float(np.clip(ci_upper, -1.0, 1.0))

    return ICCResult(
        icc=icc,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        f_value=float(f_value),
        p_value=p_value,
        label=ICCResult.reliability_label(icc),
    )


# ---------------------------------------------------------------------------
# Effect size: Cohen's d
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD) between two independent samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    pooled_sd = float(np.sqrt(pooled_var))
    if pooled_sd < 1e-15:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def cohens_d_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# Autocorrelation (lag-1)
# ---------------------------------------------------------------------------

def autocorrelation_lag1(values: np.ndarray) -> float:
    """Compute lag-1 autocorrelation of a time series.

    Values near 0 suggest random noise; values near 1 suggest systematic
    drift or correlated motion artifacts.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 4:
        return np.nan
    mean = np.mean(x)
    c0 = np.sum((x - mean) ** 2)
    if c0 < 1e-15:
        return 0.0
    c1 = np.sum((x[1:] - mean) * (x[:-1] - mean))
    return float(c1 / c0)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for a statistic, no normality assumption.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(x))

    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.array([float(stat_fn(rng.choice(x, size=n, replace=True))) for _ in range(n_bootstrap)])

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(boot, alpha * 100))
    ci_upper = float(np.percentile(boot, (1 - alpha) * 100))
    return (point, ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Normality and comparison tests
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestResult:
    """Result of a statistical hypothesis test."""
    test_name: str
    source: str
    domain: str
    trial: str
    joint: str
    statistic: float
    p_value: float
    p_corrected: float
    significant: bool
    effect_size: float
    effect_label: str
    interpretation: str


def normality_tests(
    jitter: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Shapiro-Wilk normality test per source+domain+trial+joint.

    Determines whether parametric tests are appropriate for each subset.
    """
    if jitter.empty:
        return pd.DataFrame()

    results: list[dict] = []
    groups = jitter.groupby(["source", "domain", "trial", "joint"], sort=True)
    n_tests = len(groups)

    for (source, domain, trial, joint), group in groups:
        values = group["normalized_jitter"].dropna().values
        if len(values) < 8:
            results.append({
                "source": source, "domain": domain, "trial": trial, "joint": joint,
                "n": len(values), "shapiro_stat": np.nan, "p_value": np.nan,
                "p_corrected": np.nan, "normal": None,
                "note": "Too few samples for normality test (n < 8).",
            })
            continue

        stat, p = sp_stats.shapiro(values[:5000])  # Shapiro-Wilk limit
        p_corrected = min(float(p) * n_tests, 1.0)  # Bonferroni
        results.append({
            "source": source, "domain": domain, "trial": trial, "joint": joint,
            "n": len(values), "shapiro_stat": float(stat), "p_value": float(p),
            "p_corrected": p_corrected, "normal": p_corrected > alpha,
            "note": "Normal" if p_corrected > alpha else "Non-normal (use non-parametric tests).",
        })

    return pd.DataFrame(results)


def cross_domain_tests(
    jitter: pd.DataFrame,
    alpha: float = 0.05,
) -> list[TestResult]:
    """Compare normalized jitter between sources for each trial+joint.

    Runs Mann-Whitney U (non-parametric) and Levene's test for variance equality.
    Applies Bonferroni correction across all comparisons.
    """
    if jitter.empty:
        return []

    sources = sorted(jitter["source"].unique())
    if len(sources) < 2:
        return []

    comparisons: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
    for (trial, joint), group in jitter.groupby(["trial", "joint"]):
        by_source = {}
        for source, sgroup in group.groupby("source"):
            vals = sgroup["normalized_jitter"].dropna().values
            if len(vals) >= 3:
                by_source[source] = vals
        if len(by_source) >= 2:
            src_list = sorted(by_source.keys())
            comparisons.append((str(trial), str(joint), f"{src_list[0]}_vs_{src_list[1]}",
                                by_source[src_list[0]], by_source[src_list[1]]))

    n_tests = len(comparisons) * 2  # Mann-Whitney + Levene for each
    results: list[TestResult] = []

    for trial, joint, label, a, b in comparisons:
        # Mann-Whitney U
        try:
            u_stat, u_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u_stat, u_p = np.nan, np.nan
        u_p_corr = min(float(u_p) * n_tests, 1.0) if np.isfinite(u_p) else np.nan
        d = cohens_d(a, b)
        results.append(TestResult(
            test_name="Mann-Whitney U",
            source=label, domain="", trial=trial, joint=joint,
            statistic=float(u_stat), p_value=float(u_p),
            p_corrected=u_p_corr,
            significant=u_p_corr < alpha if np.isfinite(u_p_corr) else False,
            effect_size=d,
            effect_label=cohens_d_label(d),
            interpretation=f"{'Significant' if u_p_corr < alpha else 'No significant'} difference in median normalized jitter (d={d:.3f}, {cohens_d_label(d)}).",
        ))

        # Levene's test for variance equality
        try:
            lev_stat, lev_p = sp_stats.levene(a, b, center="median")
        except ValueError:
            lev_stat, lev_p = np.nan, np.nan
        lev_p_corr = min(float(lev_p) * n_tests, 1.0) if np.isfinite(lev_p) else np.nan
        var_ratio = float(np.var(a, ddof=1) / np.var(b, ddof=1)) if np.var(b, ddof=1) > 1e-15 else np.nan
        results.append(TestResult(
            test_name="Levene's (variance equality)",
            source=label, domain="", trial=trial, joint=joint,
            statistic=float(lev_stat), p_value=float(lev_p),
            p_corrected=lev_p_corr,
            significant=lev_p_corr < alpha if np.isfinite(lev_p_corr) else False,
            effect_size=var_ratio if np.isfinite(var_ratio) else np.nan,
            effect_label=f"variance ratio {var_ratio:.2f}" if np.isfinite(var_ratio) else "N/A",
            interpretation=f"{'Significant' if lev_p_corr < alpha else 'No significant'} difference in jitter variance (ratio={var_ratio:.2f})." if np.isfinite(var_ratio) else "Variance comparison not available.",
        ))

    return results


# ---------------------------------------------------------------------------
# Per-joint reliability summary
# ---------------------------------------------------------------------------

def joint_reliability(
    jitter: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-joint reliability table: CV, autocorrelation, bootstrap CIs.

    This is the table a thesis examiner will scan to decide which joints
    are reliable enough for cross-domain claims.
    """
    if summary.empty or jitter.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for (source, domain, trial, joint), group in jitter.groupby(
        ["source", "domain", "trial", "joint"], sort=True
    ):
        values = group["normalized_jitter"].dropna().values
        if len(values) < 3:
            continue
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1))
        cv = std_val / mean_val if mean_val > 1e-15 else np.nan
        ac1 = autocorrelation_lag1(values)
        pt, ci_lo, ci_hi = bootstrap_ci(values, statistic="mean")

        rows.append({
            "source": source, "domain": domain, "trial": trial, "joint": joint,
            "n": len(values),
            "mean": mean_val,
            "std": std_val,
            "cv": cv,
            "autocorr_lag1": ac1,
            "ci_lower_95": ci_lo,
            "ci_upper_95": ci_hi,
            "reliability_note": _joint_note(cv, ac1),
        })

    return pd.DataFrame(rows)


def _joint_note(cv: float, ac1: float) -> str:
    notes = []
    if np.isfinite(cv):
        if cv > 1.0:
            notes.append("High CV — noisy landmark")
        elif cv < 0.3:
            notes.append("Low CV — stable measurement")
    if np.isfinite(ac1):
        if ac1 > 0.5:
            notes.append("high autocorrelation — systematic pattern")
        elif ac1 < 0.1:
            notes.append("near-zero autocorrelation — random noise")
    return "; ".join(notes) if notes else "—"


# ---------------------------------------------------------------------------
# Full statistical report builder
# ---------------------------------------------------------------------------

def run_full_stats(
    jitter: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run all statistical analyses and return named DataFrames for reporting."""
    output: dict[str, pd.DataFrame] = {}

    # Normality
    output["normality"] = normality_tests(jitter)

    # Cross-domain tests
    test_results = cross_domain_tests(jitter)
    if test_results:
        output["cross_domain_tests"] = pd.DataFrame([
            {
                "test": r.test_name,
                "trial": r.trial,
                "joint": r.joint,
                "comparison": r.source,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "p_corrected": r.p_corrected,
                "significant": r.significant,
                "effect_size": r.effect_size,
                "effect_label": r.effect_label,
                "interpretation": r.interpretation,
            }
            for r in test_results
        ])

    # Bland-Altman from frame-level jitter
    ba_results, ba_table = bland_altman_from_jitter(jitter)
    if not ba_table.empty:
        output["bland_altman"] = ba_table

    # Joint reliability
    reliability = joint_reliability(jitter, summary)
    if not reliability.empty:
        output["joint_reliability"] = reliability

    return output
