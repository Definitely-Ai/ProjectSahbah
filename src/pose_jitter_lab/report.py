"""Generate a self-contained thesis-grade HTML report with embedded charts.

The report is designed to be opened locally, printed to PDF, or presented
in a thesis defense. Every chart is publication-ready with proper axis labels,
units, and consistent styling.
"""

from __future__ import annotations

from base64 import b64encode
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, safe for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .stats import (
    BlandAltmanResult,
    bland_altman_from_jitter,
    bootstrap_ci,
    cohens_d,
    cohens_d_label,
    compute_icc,
    cross_domain_tests,
    joint_reliability,
    normality_tests,
    run_full_stats,
)
from .quality import (
    flag_outliers,
    quality_scorecard,
    shoulder_reliability,
    convergence_analysis,
)


# ── Global matplotlib styling ──────────────────────────────────────────────

FONT_FAMILY = "sans-serif"
CHART_DPI = 200
STYLE_RC = {
    "font.family": FONT_FAMILY,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": CHART_DPI,
    "savefig.dpi": CHART_DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
}

# Color palette
TEAL = "#0f766e"
TEAL_LIGHT = "#99f6e4"
ORANGE = "#c2410c"
ORANGE_LIGHT = "#fed7aa"
PURPLE = "#6d28d9"
PURPLE_LIGHT = "#ddd6fe"
SLATE = "#334155"
SLATE_LIGHT = "#94a3b8"
GREEN = "#15803d"
AMBER = "#b45309"
RED = "#dc2626"
BLUE = "#2563eb"

SOURCE_PALETTE = {
    "mediapipe": TEAL,
    "stereo": ORANGE,
    "opencv": ORANGE,
}


def source_color(source: str) -> str:
    s = source.lower()
    for key, color in SOURCE_PALETTE.items():
        if key in s:
            return color
    return PURPLE


def source_color_light(source: str) -> str:
    s = source.lower()
    if "mediapipe" in s:
        return TEAL_LIGHT
    if "stereo" in s or "opencv" in s:
        return ORANGE_LIGHT
    return PURPLE_LIGHT


# ── Public API ─────────────────────────────────────────────────────────────

def generate_report(
    *,
    pose: pd.DataFrame,
    jitter: pd.DataFrame,
    summary: pd.DataFrame,
    shoulder_widths: pd.DataFrame,
    output_path: str | Path,
    title: str = "Pose Jitter Lab Report",
    shoulder_mode: str = "trial_median",
) -> Path:
    """Generate a self-contained HTML report with embedded charts."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE_RC):
        # Run statistical analyses
        stats_tables = run_full_stats(jitter, summary)
        ba_results, ba_table = bland_altman_from_jitter(jitter)
        normality = normality_tests(jitter)
        reliability = joint_reliability(jitter, summary)
        shoulder_rel = shoulder_reliability(shoulder_widths)
        quality_scores = quality_scorecard(jitter, shoulder_widths)

        # Build charts
        charts = [
            ("Normalized Jitter Heatmap", plot_joint_heatmap(summary)),
            ("Normalized Jitter by Joint", plot_normalized_by_joint(summary)),
            ("Jitter Distribution by Source", plot_distribution(jitter)),
            ("Temporal Jitter Trace", plot_temporal_trace(jitter)),
            ("Shoulder Width Stability", plot_shoulder_widths(shoulder_widths)),
        ]

        # Add Bland-Altman if we have cross-source data
        if ba_results:
            charts.append(("Bland-Altman Agreement", plot_bland_altman(ba_results)))

        # Add CDF overlay
        cdf_html = plot_cdf_overlay(jitter)
        if cdf_html:
            charts.append(("Cumulative Distribution", cdf_html))

        # Add radar chart
        radar_html = plot_radar(summary)
        if radar_html:
            charts.append(("Per-Joint Jitter Fingerprint", radar_html))

        # Add correlation scatter
        corr_html = plot_correlation_scatter(jitter)
        if corr_html:
            charts.append(("Cross-Domain Correlation", corr_html))

        html = render_html(
            title=title,
            pose=pose,
            jitter=jitter,
            summary=summary,
            shoulder_widths=shoulder_widths,
            charts=charts,
            shoulder_mode=shoulder_mode,
            stats_tables=stats_tables,
            ba_results=ba_results,
            ba_table=ba_table,
            normality=normality,
            reliability=reliability,
            shoulder_rel=shoulder_rel,
            quality_scores=quality_scores,
        )

    out.write_text(html, encoding="utf-8")
    return out


# ── HTML Template ──────────────────────────────────────────────────────────

def render_html(
    *,
    title: str,
    pose: pd.DataFrame,
    jitter: pd.DataFrame,
    summary: pd.DataFrame,
    shoulder_widths: pd.DataFrame,
    charts: list[tuple[str, str]],
    shoulder_mode: str,
    stats_tables: dict[str, pd.DataFrame],
    ba_results: list[BlandAltmanResult],
    ba_table: pd.DataFrame,
    normality: pd.DataFrame,
    reliability: pd.DataFrame,
    shoulder_rel: pd.DataFrame,
    quality_scores: list,
) -> str:
    overview = build_overview(pose, jitter, shoulder_widths)
    insights = build_insight_cards(summary, jitter, shoulder_widths)
    executive = build_executive_summary(summary, jitter, ba_results, quality_scores)
    comparison_table = source_comparison_table(summary)
    quality_table = quality_gate_table(jitter, shoulder_widths)
    top_table = top_joints_table(summary)
    stats_section = build_stats_section(stats_tables, ba_table, normality, reliability, shoulder_rel)
    quality_section = build_quality_section(quality_scores)
    chart_blocks = "\n".join(
        f"""
        <section class="panel chart-panel">
          <h2>{escape(chart_title)}</h2>
          {image_html}
        </section>
        """
        for chart_title, image_html in charts
    )
    downloads = output_links()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  {_css()}
</head>
<body>
  <nav class="sidebar" id="sidebar">
    <div class="sidebar-title">Navigation</div>
    <a href="#overview">Overview</a>
    <a href="#executive">Executive Summary</a>
    <a href="#comparison">Source Comparison</a>
    <a href="#statistics">Statistical Analysis</a>
    <a href="#quality-gates">Quality Gates</a>
    <a href="#data-quality">Data Quality</a>
    <a href="#top-joints">Top Jitter Joints</a>
    <a href="#charts">Visualizations</a>
    <a href="#downloads">Downloads</a>
    <button class="theme-toggle" onclick="toggleTheme()" title="Toggle dark mode">◑</button>
  </nav>
  <div class="main-wrapper">
  <header>
    <button class="menu-btn" onclick="toggleSidebar()">☰</button>
    <h1>{escape(title)}</h1>
    <p>Cross-domain pose jitter analysis &mdash; normalized 2D landmarks vs stereo millimeter reconstructions.</p>
    <p class="method">Jitter = frame-to-frame Euclidean landmark displacement &divide; same-domain shoulder width. Shoulder mode: <code>{escape(shoulder_mode)}</code>.</p>
  </header>
  <main>
    <section id="overview" class="grid">
      {overview}
    </section>

    <section id="executive" class="panel feature executive">
      <h2>Executive Summary</h2>
      {executive}
    </section>

    <section class="insights">
      {insights}
    </section>

    <section id="comparison" class="panel feature">
      <h2>Source Comparison</h2>
      {comparison_table}
    </section>

    {stats_section}

    <section id="quality-gates" class="panel">
      <h2>Quality Gates</h2>
      {quality_table}
    </section>

    {quality_section}

    <section id="top-joints" class="panel">
      <h2>Highest Normalized Jitter</h2>
      {top_table}
    </section>

    <div id="charts">
    {chart_blocks}
    </div>

    <section id="downloads" class="panel">
      <h2>Interpretation Guardrails</h2>
      <p>Compare normalized statistics across domains, not raw coordinate values. Use the raw values to explain engineering behavior inside each system. If the shoulder-width trace has spikes or missing frames, review those frames before making statistical claims.</p>
      {downloads}
    </section>
    <footer>Generated by Pose Jitter Lab &mdash; thesis-grade analysis toolkit.</footer>
  </main>
  </div>
  {_js()}
</body>
</html>
"""


# ── CSS ────────────────────────────────────────────────────────────────────

def _css() -> str:
    return """<style>
    :root {
      --bg: #f4f5f9;
      --bg-alt: #ffffff;
      --ink: #17202a;
      --ink-2: #334155;
      --muted: #5c6672;
      --line: #dbe1ea;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-bg: rgba(15, 118, 110, 0.08);
      --accent-2: #c2410c;
      --accent-3: #6d28d9;
      --good: #15803d;
      --warn: #b45309;
      --danger: #dc2626;
      --shadow: 0 4px 24px rgba(23, 32, 42, 0.07);
      --shadow-lg: 0 18px 45px rgba(23, 32, 42, 0.10);
      --radius: 10px;
      --sidebar-w: 220px;
    }
    [data-theme="dark"] {
      --bg: #0f172a;
      --bg-alt: #1e293b;
      --ink: #e2e8f0;
      --ink-2: #cbd5e1;
      --muted: #94a3b8;
      --line: #334155;
      --panel: #1e293b;
      --accent-bg: rgba(15, 118, 110, 0.18);
      --shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 18px 45px rgba(0, 0, 0, 0.4);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", Inter, system-ui, -apple-system, sans-serif;
      line-height: 1.55;
    }

    /* Sidebar */
    .sidebar {
      position: fixed;
      top: 0; left: 0; bottom: 0;
      width: var(--sidebar-w);
      background: var(--panel);
      border-right: 1px solid var(--line);
      padding: 22px 16px;
      z-index: 100;
      overflow-y: auto;
      transition: transform 0.25s ease;
    }
    .sidebar-title {
      font-weight: 700;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 16px;
    }
    .sidebar a {
      display: block;
      padding: 7px 12px;
      margin: 2px 0;
      border-radius: 6px;
      color: var(--ink-2);
      text-decoration: none;
      font-size: 0.88rem;
      font-weight: 500;
      transition: background 0.15s, color 0.15s;
    }
    .sidebar a:hover { background: var(--accent-bg); color: var(--accent); }
    .theme-toggle {
      position: absolute;
      bottom: 16px;
      left: 16px;
      width: 38px; height: 38px;
      border-radius: 50%;
      border: 1px solid var(--line);
      background: var(--bg);
      color: var(--ink);
      font-size: 1.2rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .theme-toggle:hover { background: var(--accent-bg); border-color: var(--accent); }
    .menu-btn {
      display: none;
      position: absolute;
      top: 14px; left: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 1.2rem;
      cursor: pointer;
      z-index: 101;
    }

    .main-wrapper {
      margin-left: var(--sidebar-w);
      transition: margin-left 0.25s ease;
    }

    header {
      position: relative;
      padding: 48px 48px 34px;
      background:
        linear-gradient(135deg, rgba(15,118,110,0.10), rgba(109,40,217,0.06) 45%, rgba(194,65,12,0.08)),
        var(--panel);
      border-bottom: 1px solid var(--line);
      overflow: hidden;
    }
    header::after {
      content: "";
      position: absolute;
      right: -8%; top: -54%;
      width: 42%; height: 180%;
      background:
        linear-gradient(120deg, transparent 0 16%, rgba(15,118,110,0.08) 16% 31%, transparent 31% 42%, rgba(194,65,12,0.08) 42% 56%, transparent 56% 100%);
      transform: rotate(-7deg);
      pointer-events: none;
    }
    header > * { position: relative; z-index: 1; }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 28px 52px;
    }
    h1 {
      margin: 0 0 10px;
      max-width: 920px;
      font-size: clamp(1.8rem, 3.5vw, 3rem);
      letter-spacing: -0.02em;
      line-height: 1.08;
    }
    h2 {
      margin: 0 0 16px;
      font-size: 1.2rem;
      letter-spacing: -0.01em;
      color: var(--ink-2);
    }
    p { margin: 0; color: var(--muted); max-width: 880px; }
    .method {
      margin-top: 18px;
      max-width: 860px;
      padding: 14px 18px;
      border: 1px solid rgba(15,118,110,0.20);
      border-left: 4px solid var(--accent);
      border-radius: var(--radius);
      background: var(--accent-bg);
      color: var(--ink);
      font-size: 0.95rem;
    }

    /* Grid layouts */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: 14px;
      margin: 24px 0;
    }
    .insights {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin: 18px 0 24px;
    }

    /* Cards */
    .stat, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: 0 1px 3px rgba(23,32,42,0.04);
    }
    .stat { padding: 18px; }
    .stat strong {
      display: block;
      font-size: 1.7rem;
      line-height: 1;
      color: var(--accent);
      font-variant-numeric: tabular-nums;
    }
    .stat span {
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .insight {
      position: relative;
      min-height: 120px;
      padding: 18px;
      overflow: hidden;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
    }
    .insight::before {
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      background: linear-gradient(90deg, var(--accent), var(--accent-3), var(--accent-2));
    }
    .insight b {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .insight strong {
      display: block;
      color: var(--ink);
      font-size: 1.25rem;
      line-height: 1.18;
    }
    .insight span {
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.88rem;
    }

    .panel {
      padding: 22px;
      margin: 18px 0;
      overflow-x: auto;
    }
    .panel.feature {
      border-top: 4px solid var(--accent);
      box-shadow: var(--shadow-lg);
    }
    .panel.executive {
      background: linear-gradient(135deg, var(--panel), var(--accent-bg));
    }
    .chart-panel img {
      border-radius: 6px;
      margin-top: 4px;
    }

    img {
      max-width: 100%;
      height: auto;
      display: block;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th, td {
      text-align: left;
      padding: 9px 12px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      background: var(--bg);
      position: sticky;
      top: 0;
    }
    code {
      background: var(--accent-bg);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.88em;
    }
    .badge {
      display: inline-block;
      min-width: 68px;
      padding: 3px 9px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 700;
      text-align: center;
    }
    .pass { color: #fff; background: var(--good); }
    .review { color: #fff; background: var(--warn); }
    .fail { color: #fff; background: var(--danger); }
    .grade-a { color: #fff; background: var(--good); }
    .grade-b { color: #fff; background: #0891b2; }
    .grade-c { color: #fff; background: var(--warn); }
    .grade-d { color: #fff; background: var(--danger); }
    .downloads {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    .downloads a {
      color: #ffffff;
      text-decoration: none;
      background: var(--ink);
      border-radius: 6px;
      padding: 9px 14px;
      font-size: 0.88rem;
      font-weight: 650;
      transition: background 0.15s;
    }
    .downloads a:hover { background: var(--accent); }
    .exec-finding {
      padding: 10px 0;
      border-bottom: 1px solid var(--line);
      line-height: 1.6;
      color: var(--ink);
    }
    .exec-finding:last-child { border-bottom: none; }
    .exec-finding strong { color: var(--accent); }

    footer {
      color: var(--muted);
      padding-top: 18px;
      font-size: 0.85rem;
      text-align: center;
    }

    @media (max-width: 900px) {
      .sidebar { transform: translateX(-100%); }
      .sidebar.open { transform: translateX(0); }
      .main-wrapper { margin-left: 0; }
      .menu-btn { display: block; }
      header { padding: 36px 20px 24px; }
      main { padding: 16px 14px 34px; }
    }
    @media print {
      .sidebar, .menu-btn, .theme-toggle, .downloads { display: none !important; }
      .main-wrapper { margin-left: 0 !important; }
      body { background: #fff; color: #000; font-size: 10pt; }
      header { background: #fff !important; border-bottom: 2px solid #000; padding: 20px; }
      header::after { display: none; }
      .panel, .stat, .insight { box-shadow: none; border: 1px solid #ccc; break-inside: avoid; }
      .panel.feature { border-top: 3px solid #000; }
      h1 { font-size: 18pt; }
      h2 { font-size: 12pt; }
      img { max-width: 100%; }
      table { font-size: 8pt; }
      main { max-width: 100%; padding: 10px; }
    }
  </style>"""


def _js() -> str:
    return """<script>
    function toggleTheme() {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme');
      html.setAttribute('data-theme', current === 'dark' ? '' : 'dark');
    }
    function toggleSidebar() {
      document.getElementById('sidebar').classList.toggle('open');
    }
    // Smooth scroll for nav
    document.querySelectorAll('.sidebar a').forEach(a => {
      a.addEventListener('click', e => {
        const sidebar = document.getElementById('sidebar');
        if (sidebar.classList.contains('open')) sidebar.classList.remove('open');
      });
    });
  </script>"""


# ── Section builders ───────────────────────────────────────────────────────

def build_overview(pose: pd.DataFrame, jitter: pd.DataFrame, shoulder_widths: pd.DataFrame) -> str:
    stats = {
        "Pose rows": f"{len(pose):,}",
        "Frame pairs": f"{len(jitter):,}",
        "Trials": pose["trial"].nunique() if "trial" in pose else 0,
        "Joints": pose["joint"].nunique() if "joint" in pose else 0,
        "Sources": pose["source"].nunique() if "source" in pose else 0,
        "Shoulder frames": f"{shoulder_widths['shoulder_width'].notna().sum():,}" if not shoulder_widths.empty else 0,
    }
    return "\n".join(
        f'<div class="stat"><strong>{value}</strong><span>{escape(str(label))}</span></div>'
        for label, value in stats.items()
    )


def build_executive_summary(
    summary: pd.DataFrame,
    jitter: pd.DataFrame,
    ba_results: list[BlandAltmanResult],
    quality_scores: list,
) -> str:
    """Auto-generate a paragraph interpreting key findings."""
    findings: list[str] = []

    if not summary.empty and summary["normalized_mean"].notna().any():
        sources = summary["source"].unique()
        if len(sources) >= 2:
            source_means = summary.groupby("source")["normalized_mean"].mean()
            sorted_sources = source_means.sort_values(ascending=False)
            hi_src, hi_val = sorted_sources.index[0], sorted_sources.iloc[0]
            lo_src, lo_val = sorted_sources.index[-1], sorted_sources.iloc[-1]
            ratio = hi_val / lo_val if lo_val > 1e-12 else float("inf")
            findings.append(
                f"<strong>{escape(hi_src)}</strong> exhibited {ratio:.1f}× higher mean normalized jitter "
                f"than <strong>{escape(lo_src)}</strong> ({hi_val:.4g} vs {lo_val:.4g} shoulder widths/frame)."
            )

        top = summary.sort_values("normalized_mean", ascending=False).iloc[0]
        findings.append(
            f"The highest per-joint jitter was observed at <strong>{escape(str(top['joint']))}</strong> "
            f"({escape(str(top['source']))}) with a mean of {top['normalized_mean']:.4g} shoulder widths/frame."
        )

    if ba_results:
        good = sum(1 for r in ba_results if r.percent_within_loa >= 90)
        findings.append(
            f"Bland-Altman analysis: {good}/{len(ba_results)} joint comparisons showed "
            f"≥90% of values within limits of agreement."
        )

    if quality_scores:
        grades = [q.overall_grade for q in quality_scores]
        a_count = grades.count("A")
        findings.append(
            f"Data quality: {a_count}/{len(grades)} source-trial combinations received grade A."
        )

    if not findings:
        findings.append("Add data from at least two sources to generate cross-domain findings.")

    return "\n".join(f'<div class="exec-finding">{f}</div>' for f in findings)


def build_insight_cards(summary: pd.DataFrame, jitter: pd.DataFrame, shoulder_widths: pd.DataFrame) -> str:
    valid_rate = float(jitter["normalized_jitter"].notna().mean() * 100) if not jitter.empty else 0.0
    highest = "No valid jitter"
    highest_note = "Add at least two frames per joint."
    if not summary.empty and summary["normalized_mean"].notna().any():
        top = summary.sort_values("normalized_mean", ascending=False).iloc[0]
        highest = f"{top['joint']} | {top['source']}"
        highest_note = f"Mean normalized jitter {top['normalized_mean']:.4g} shoulder widths per frame."

    ratio_text, ratio_note = strongest_source_gap(summary)
    stability_text, stability_note = shoulder_stability_text(shoulder_widths)
    cards = [
        ("Data Integrity", f"{valid_rate:.1f}% valid", "Normalized jitter rows with a valid shoulder denominator."),
        ("Peak Finding", highest, highest_note),
        ("Cross-Domain Gap", ratio_text, ratio_note),
        ("Shoulder Stability", stability_text, stability_note),
    ]
    return "\n".join(
        f"""
        <div class="insight">
          <b>{escape(label)}</b>
          <strong>{escape(value)}</strong>
          <span>{escape(note)}</span>
        </div>
        """
        for label, value, note in cards
    )


def strongest_source_gap(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "Not available", "Need at least two sources with matched joints."
    pivot = summary.pivot_table(
        index=["trial", "joint"],
        columns="source",
        values="normalized_mean",
        aggfunc="first",
    )
    non_scale_joints = ~pivot.index.get_level_values("joint").isin(["left_shoulder", "right_shoulder"])
    focused = pivot[non_scale_joints]
    if not focused.empty:
        pivot = focused
    if pivot.shape[1] < 2:
        return "Not available", "Need at least two sources with matched joints."

    best: tuple[float, str, str, str, str, float, float] | None = None
    for _, row in pivot.dropna(how="any").iterrows():
        values = row.astype(float)
        if (values <= 0).any():
            continue
        high_source = str(values.idxmax())
        low_source = str(values.idxmin())
        ratio = float(values.max() / values.min())
        joint = str(row.name[1])
        trial = str(row.name[0])
        candidate = (ratio, joint, trial, high_source, low_source, float(values.max()), float(values.min()))
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return "Not available", "Matched positive values were not available."
    ratio, joint, trial, high_source, low_source, high_value, low_value = best
    return f"{ratio:.2f}x on {joint}", f"{high_source} exceeded {low_source} in {trial} ({high_value:.4g} vs {low_value:.4g})."


def shoulder_stability_text(shoulder_widths: pd.DataFrame) -> tuple[str, str]:
    if shoulder_widths.empty or shoulder_widths["shoulder_width"].dropna().empty:
        return "No shoulders", "Shoulder landmarks are required for normalized jitter."
    grouped = shoulder_widths.dropna(subset=["shoulder_width"]).groupby(["source", "domain", "trial"])
    cvs = grouped["shoulder_width"].agg(lambda values: float(values.std() / values.mean()) if values.mean() else np.nan)
    worst = cvs.dropna().sort_values(ascending=False)
    if worst.empty:
        return "No estimate", "Shoulder width variability could not be estimated."
    key = worst.index[0]
    return f"{worst.iloc[0] * 100:.2f}% max CV", f"Worst trace: {key[0]} | {key[1]} | {key[2]}."


def source_comparison_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "<p>No source comparison was available.</p>"
    pivot = summary.pivot_table(
        index=["trial", "joint"],
        columns=["source", "domain"],
        values="normalized_mean",
        aggfunc="first",
    )
    if pivot.empty:
        return "<p>No matched normalized jitter values were available.</p>"
    pivot.columns = [f"{source} | {domain}" for source, domain in pivot.columns]
    pivot = pivot.reset_index()
    value_columns = [column for column in pivot.columns if column not in {"trial", "joint"}]
    pivot["highest_mean"] = pivot[value_columns].idxmax(axis=1)
    pivot = pivot.sort_values(value_columns, ascending=False).head(12)
    for column in value_columns:
        pivot[column] = pivot[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return pivot.to_html(index=False, escape=True)


def quality_gate_table(jitter: pd.DataFrame, shoulder_widths: pd.DataFrame) -> str:
    valid_count = int(jitter["normalized_jitter"].notna().sum()) if not jitter.empty else 0
    total_count = int(len(jitter))
    missing_shoulder = int(shoulder_widths["shoulder_width"].isna().sum()) if not shoulder_widths.empty else 0
    max_cv = np.nan
    if not shoulder_widths.empty and shoulder_widths["shoulder_width"].notna().any():
        max_cv = (
            shoulder_widths.dropna(subset=["shoulder_width"])
            .groupby(["source", "domain", "trial"])["shoulder_width"]
            .agg(lambda values: float(values.std() / values.mean()) if values.mean() else np.nan)
            .max()
        )

    rows = [
        {
            "check": "Normalized jitter denominator",
            "status": "pass" if total_count and valid_count == total_count else "review",
            "result": f"{valid_count} of {total_count} rows valid",
        },
        {
            "check": "Shoulder frames",
            "status": "pass" if missing_shoulder == 0 and not shoulder_widths.empty else "review",
            "result": f"{missing_shoulder} missing shoulder-width frames",
        },
        {
            "check": "Shoulder-width stability",
            "status": "pass" if pd.notna(max_cv) and max_cv < 0.05 else "review",
            "result": "not available" if pd.isna(max_cv) else f"max CV {max_cv * 100:.2f}%",
        },
    ]
    table = pd.DataFrame(rows)
    table["status"] = table["status"].map(lambda value: f'<span class="badge {value}">{value.upper()}</span>')
    return table.to_html(index=False, escape=False)


def top_joints_table(summary: pd.DataFrame, limit: int = 12) -> str:
    if summary.empty:
        return "<p>No jitter rows were available to summarize.</p>"
    columns = [
        "source",
        "domain",
        "trial",
        "joint",
        "frames",
        "normalized_mean",
        "normalized_std",
        "normalized_p95",
        "normalized_max",
    ]
    table = (
        summary.sort_values("normalized_mean", ascending=False)
        .loc[:, columns]
        .head(limit)
        .copy()
    )
    for column in table.select_dtypes(include="number").columns:
        if column != "frames":
            table[column] = table[column].map(lambda value: f"{value:.6g}")
    return table.to_html(index=False, escape=True)


def build_stats_section(
    stats_tables: dict[str, pd.DataFrame],
    ba_table: pd.DataFrame,
    normality: pd.DataFrame,
    reliability: pd.DataFrame,
    shoulder_rel: pd.DataFrame,
) -> str:
    sections: list[str] = []

    # Bland-Altman summary
    if not ba_table.empty:
        fmt_ba = ba_table.copy()
        for col in ["bias", "std_diff", "loa_lower", "loa_upper"]:
            if col in fmt_ba.columns:
                fmt_ba[col] = fmt_ba[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        if "pct_within_loa" in fmt_ba.columns:
            fmt_ba["pct_within_loa"] = fmt_ba["pct_within_loa"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")
        sections.append(f"""
        <section class="panel">
          <h2>Bland-Altman Agreement</h2>
          <p style="margin-bottom:12px">Method comparison between sources. Bias near zero and ≥95% within LoA indicates excellent agreement.</p>
          {fmt_ba.to_html(index=False, escape=True)}
        </section>
        """)

    # Cross-domain tests
    if "cross_domain_tests" in stats_tables and not stats_tables["cross_domain_tests"].empty:
        cdt = stats_tables["cross_domain_tests"].copy()
        for col in ["statistic", "p_value", "p_corrected", "effect_size"]:
            if col in cdt.columns:
                cdt[col] = cdt[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        if "significant" in cdt.columns:
            cdt["significant"] = cdt["significant"].map(
                lambda v: '<span class="badge review">YES</span>' if v else '<span class="badge pass">NO</span>'
            )
        sections.append(f"""
        <section class="panel">
          <h2>Statistical Tests</h2>
          <p style="margin-bottom:12px">Mann-Whitney U for median comparison, Levene's for variance equality. P-values are Bonferroni-corrected.</p>
          {cdt.to_html(index=False, escape=False)}
        </section>
        """)

    # Normality
    if not normality.empty:
        fmt_norm = normality.copy()
        for col in ["shapiro_stat", "p_value", "p_corrected"]:
            if col in fmt_norm.columns:
                fmt_norm[col] = fmt_norm[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        sections.append(f"""
        <section class="panel">
          <h2>Normality Tests (Shapiro-Wilk)</h2>
          <p style="margin-bottom:12px">Determines whether parametric tests are appropriate. P-values are Bonferroni-corrected.</p>
          {fmt_norm.to_html(index=False, escape=True)}
        </section>
        """)

    # Joint reliability
    if not reliability.empty:
        fmt_rel = reliability.copy()
        for col in ["mean", "std", "cv", "autocorr_lag1", "ci_lower_95", "ci_upper_95"]:
            if col in fmt_rel.columns:
                fmt_rel[col] = fmt_rel[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        sections.append(f"""
        <section class="panel">
          <h2>Per-Joint Reliability</h2>
          <p style="margin-bottom:12px">CV, autocorrelation, and bootstrap 95% CIs. Low CV + near-zero autocorrelation = random noise (good). High CV or high autocorrelation warrants review.</p>
          {fmt_rel.to_html(index=False, escape=True)}
        </section>
        """)

    # Shoulder reliability
    if not shoulder_rel.empty:
        fmt_sr = shoulder_rel.copy()
        for col in ["mean_width", "median_width", "std_width", "cv", "spike_pct", "max_pct_change"]:
            if col in fmt_sr.columns:
                fmt_sr[col] = fmt_sr[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "—")
        if "reliability" in fmt_sr.columns:
            fmt_sr["reliability"] = fmt_sr["reliability"].map(lambda v: f'<span class="badge {"pass" if v in ("excellent","good") else "review"}">{v.upper()}</span>')
        sections.append(f"""
        <section class="panel">
          <h2>Shoulder Landmark Reliability</h2>
          <p style="margin-bottom:12px">Shoulder width is the normalization denominator — its stability directly affects all normalized jitter values.</p>
          {fmt_sr.to_html(index=False, escape=False)}
        </section>
        """)

    if sections:
        return f'<div id="statistics">{"".join(sections)}</div>'
    return ""


def build_quality_section(quality_scores: list) -> str:
    if not quality_scores:
        return ""
    rows = []
    for q in quality_scores:
        grade_cls = f"grade-{q.overall_grade.lower()}"
        rows.append({
            "source": q.source,
            "trial": q.trial,
            "completeness": f"{q.completeness_pct:.1f}%",
            "continuity": f"{q.continuity_pct:.1f}%",
            "shoulder": q.shoulder_reliability,
            "outliers": f"{q.outlier_pct:.1f}%",
            "frames": q.n_frames,
            "grade": f'<span class="badge {grade_cls}">{q.overall_grade}</span>',
            "notes": "; ".join(q.notes) if q.notes else "—",
        })
    df = pd.DataFrame(rows)
    return f"""
    <section id="data-quality" class="panel">
      <h2>Data Quality Scorecard</h2>
      <p style="margin-bottom:12px">Per-trial quality assessment. Grade A requires ≥98% completeness, ≥98% continuity, good shoulder stability, and &lt;3% outliers.</p>
      {df.to_html(index=False, escape=False)}
    </section>
    """


def output_links() -> str:
    files = [
        ("Frame jitter CSV", "jitter_frames.csv"),
        ("Summary CSV", "jitter_summary.csv"),
        ("Comparison CSV", "comparison.csv"),
        ("Shoulder widths CSV", "shoulder_widths.csv"),
    ]
    links = "\n".join(f'<a href="{escape(path)}">{escape(label)}</a>' for label, path in files)
    return f'<div class="downloads">{links}</div>'


# ── Chart functions ────────────────────────────────────────────────────────

def plot_joint_heatmap(summary: pd.DataFrame) -> str:
    if summary.empty:
        return empty_chart("No summary data")
    plot_data = summary.copy()
    plot_data["series"] = plot_data["source"] + " | " + plot_data["domain"]
    pivot = (
        plot_data.pivot_table(index="joint", columns="series", values="normalized_mean", aggfunc="first")
        .sort_index()
    )
    if pivot.empty:
        return empty_chart("No heatmap data")

    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    threshold = float(np.nanmin(finite) + (np.nanmax(finite) - np.nanmin(finite)) * 0.45) if finite.size else 0.0

    fig, ax = plt.subplots(figsize=(10, max(4.5, len(pivot.index) * 0.45)))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="magma", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean Normalized Jitter by Joint and Source", pad=12)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if np.isfinite(value):
                color = "white" if value >= threshold else "#17202a"
                ax.text(x, y, f"{value:.3g}", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    fig.colorbar(image, ax=ax, label="Mean normalized jitter (shoulder widths/frame)")
    fig.tight_layout()
    return fig_to_image(fig)


def plot_normalized_by_joint(summary: pd.DataFrame) -> str:
    if summary.empty:
        return empty_chart("No summary data")
    plot_data = summary.copy()
    plot_data["label"] = plot_data["source"] + " | " + plot_data["joint"]
    plot_data = plot_data.sort_values("normalized_mean", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(11, max(5, len(plot_data) * 0.32)))
    colors = [source_color(source) for source in plot_data["source"]]

    bars = ax.barh(plot_data["label"], plot_data["normalized_mean"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean Normalized Jitter (shoulder widths / frame)")
    ax.set_title("Normalized Jitter Ranked by Joint", pad=12)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{width:.4g}', va='center', fontsize=8, color=SLATE)

    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_distribution(jitter: pd.DataFrame) -> str:
    if jitter.empty:
        return empty_chart("No jitter data")

    labels = []
    values = []
    colors = []
    for label, group in jitter.groupby(["source", "domain"], sort=True):
        clean = group["normalized_jitter"].dropna()
        if clean.empty:
            continue
        labels.append(f"{label[0]}\n{label[1]}")
        values.append(clean)
        colors.append(source_color(label[0]))
    if not values:
        return empty_chart("No valid normalized jitter values")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bp = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#17202a", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Normalized Jitter (shoulder widths / frame)")
    ax.set_title("Jitter Distribution by Source", pad=12)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_temporal_trace(jitter: pd.DataFrame) -> str:
    """Frame-by-frame jitter overlay showing temporal patterns."""
    if jitter.empty:
        return empty_chart("No jitter data")

    # Pick top 3 joints by mean normalized jitter
    joint_means = jitter.groupby("joint")["normalized_jitter"].mean().dropna()
    # Exclude shoulders as they are the denominator
    joint_means = joint_means[~joint_means.index.isin(["left_shoulder", "right_shoulder"])]
    top_joints = joint_means.sort_values(ascending=False).head(3).index.tolist()

    if not top_joints:
        return empty_chart("No non-shoulder joints with valid jitter")

    fig, axes = plt.subplots(len(top_joints), 1, figsize=(12, 3.5 * len(top_joints)), sharex=True)
    if len(top_joints) == 1:
        axes = [axes]

    for ax, joint_name in zip(axes, top_joints):
        joint_data = jitter[jitter["joint"] == joint_name]
        for (source, domain), group in joint_data.groupby(["source", "domain"], sort=True):
            clean = group.dropna(subset=["normalized_jitter"]).sort_values("frame_to")
            if clean.empty:
                continue
            color = source_color(source)
            ax.plot(clean["frame_to"], clean["normalized_jitter"],
                    linewidth=0.9, alpha=0.85, color=color, label=f"{source} | {domain}")

            # Rolling mean overlay
            if len(clean) >= 10:
                rolling = clean["normalized_jitter"].rolling(15, min_periods=3, center=True).mean()
                ax.plot(clean["frame_to"], rolling,
                        linewidth=2.2, alpha=0.95, color=color, linestyle="--")

        ax.set_ylabel("Norm. jitter")
        ax.set_title(f"{joint_name}", fontsize=11, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_axisbelow(True)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Temporal Jitter Trace (solid = raw, dashed = rolling mean)", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_shoulder_widths(shoulder_widths: pd.DataFrame) -> str:
    if shoulder_widths.empty:
        return empty_chart("No shoulder width data")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, group in shoulder_widths.groupby(["source", "domain", "trial"], sort=True):
        valid = group.dropna(subset=["shoulder_width"])
        if valid.empty:
            continue
        color = source_color(str(label[0]))
        ax.plot(valid["frame"], valid["shoulder_width"], linewidth=1.8, color=color,
                label=" | ".join(map(str, label)), alpha=0.85)

        # Rolling CV band
        if len(valid) >= 10:
            rolling_mean = valid["shoulder_width"].rolling(15, min_periods=3, center=True).mean()
            rolling_std = valid["shoulder_width"].rolling(15, min_periods=3, center=True).std()
            ax.fill_between(valid["frame"], rolling_mean - rolling_std, rolling_mean + rolling_std,
                            alpha=0.12, color=color)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Shoulder Width (native domain units)")
    ax.set_title("Shoulder Width Stability", pad=12)
    ax.legend(fontsize=8, loc="best")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_bland_altman(ba_results: list[BlandAltmanResult]) -> str:
    """Bland-Altman plot: difference vs mean with LoA bands."""
    plottable = [r for r in ba_results if r.n >= 3 and len(r.means) > 0]
    if not plottable:
        return empty_chart("Insufficient paired data for Bland-Altman plot")

    fig, ax = plt.subplots(figsize=(10, 6))

    for r in plottable:
        ax.scatter(r.means, r.diffs, alpha=0.5, s=18, label=f"{r.joint} ({r.trial})", edgecolors="none")

    # Use overall stats for reference lines
    all_diffs = np.concatenate([r.diffs for r in plottable])
    all_means_arr = np.concatenate([r.means for r in plottable])
    overall_mean = float(np.mean(all_diffs))
    overall_std = float(np.std(all_diffs, ddof=1))
    loa_lo = overall_mean - 1.96 * overall_std
    loa_hi = overall_mean + 1.96 * overall_std

    x_range = [float(np.min(all_means_arr)), float(np.max(all_means_arr))]
    ax.axhline(overall_mean, color=TEAL, linewidth=2, linestyle="-", label=f"Bias = {overall_mean:.4g}")
    ax.axhline(loa_hi, color=RED, linewidth=1.5, linestyle="--", label=f"+1.96 SD = {loa_hi:.4g}")
    ax.axhline(loa_lo, color=RED, linewidth=1.5, linestyle="--", label=f"−1.96 SD = {loa_lo:.4g}")
    ax.axhline(0, color=SLATE_LIGHT, linewidth=0.8, linestyle=":")
    ax.fill_between(x_range, loa_lo, loa_hi, alpha=0.06, color=RED)

    ax.set_xlabel("Mean of Two Measurements")
    ax.set_ylabel("Difference (Source A − Source B)")
    ax.set_title("Bland-Altman Agreement Plot", pad=12)
    ax.legend(fontsize=8, loc="best")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_cdf_overlay(jitter: pd.DataFrame) -> str:
    """Cumulative distribution function overlay per source."""
    if jitter.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 5.5))
    has_data = False
    for (source, domain), group in jitter.groupby(["source", "domain"], sort=True):
        clean = group["normalized_jitter"].dropna().sort_values()
        if len(clean) < 5:
            continue
        has_data = True
        y = np.arange(1, len(clean) + 1) / len(clean)
        color = source_color(source)
        ax.step(clean, y, where="post", linewidth=2, color=color,
                label=f"{source} | {domain}", alpha=0.85)

    if not has_data:
        plt.close(fig)
        return ""

    ax.set_xlabel("Normalized Jitter (shoulder widths / frame)")
    ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Cumulative Distribution of Normalized Jitter", pad=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_axisbelow(True)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig_to_image(fig)


def plot_radar(summary: pd.DataFrame) -> str:
    """Radar/spider chart comparing jitter profiles across sources."""
    if summary.empty:
        return ""

    # Exclude shoulders from radar
    radar_data = summary[~summary["joint"].isin(["left_shoulder", "right_shoulder"])].copy()
    sources = sorted(radar_data["source"].unique())
    if len(sources) < 2:
        return ""

    joints = sorted(radar_data["joint"].unique())
    if len(joints) < 3:
        return ""

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, len(joints), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for source in sources:
        src_data = radar_data[radar_data["source"] == source]
        values = []
        for joint in joints:
            joint_rows = src_data[src_data["joint"] == joint]
            if not joint_rows.empty:
                values.append(float(joint_rows["normalized_mean"].mean()))
            else:
                values.append(0.0)
        values += values[:1]
        color = source_color(source)
        ax.plot(angles, values, linewidth=2, color=color, label=source)
        ax.fill(angles, values, alpha=0.12, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([j.replace("_", " ") for j in joints], fontsize=9)
    ax.set_title("Jitter Fingerprint by Source", y=1.1, fontsize=13)
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    return fig_to_image(fig)


def plot_correlation_scatter(jitter: pd.DataFrame) -> str:
    """Scatter plot of normalized jitter: source A vs source B."""
    if jitter.empty:
        return ""

    sources = sorted(jitter["source"].unique())
    if len(sources) < 2:
        return ""

    fig, ax = plt.subplots(figsize=(8, 8))
    has_data = False

    for (trial, joint), group in jitter.groupby(["trial", "joint"]):
        pivot = group.pivot_table(
            index="frame_to",
            columns="source",
            values="normalized_jitter",
            aggfunc="first",
        ).dropna(how="any")

        if pivot.shape[1] < 2 or len(pivot) < 3:
            continue

        x = pivot[sources[0]].values
        y = pivot[sources[1]].values
        has_data = True
        ax.scatter(x, y, alpha=0.3, s=12, edgecolors="none", label=f"{joint}")

    if not has_data:
        plt.close(fig)
        return ""

    # Regression line on all pooled data
    all_points: list[tuple[np.ndarray, np.ndarray]] = []
    for (trial, joint), group in jitter.groupby(["trial", "joint"]):
        pivot = group.pivot_table(
            index="frame_to", columns="source", values="normalized_jitter", aggfunc="first"
        ).dropna(how="any")
        if pivot.shape[1] >= 2 and len(pivot) >= 3:
            all_points.append((pivot[sources[0]].values, pivot[sources[1]].values))

    if all_points:
        all_x = np.concatenate([p[0] for p in all_points])
        all_y = np.concatenate([p[1] for p in all_points])
        mask = np.isfinite(all_x) & np.isfinite(all_y)
        all_x, all_y = all_x[mask], all_y[mask]
        if len(all_x) >= 3:
            from scipy import stats as sp_stats
            slope, intercept, r_value, _, _ = sp_stats.linregress(all_x, all_y)
            x_line = np.linspace(all_x.min(), all_x.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color=RED, linewidth=2,
                    label=f"R² = {r_value**2:.3f}")

    # Identity line
    lims = [ax.get_xlim(), ax.get_ylim()]
    low = min(lims[0][0], lims[1][0])
    high = max(lims[0][1], lims[1][1])
    ax.plot([low, high], [low, high], color=SLATE_LIGHT, linewidth=1, linestyle=":", label="Identity (y=x)")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    ax.set_xlabel(f"Normalized Jitter — {sources[0]}")
    ax.set_ylabel(f"Normalized Jitter — {sources[1]}")
    ax.set_title("Cross-Domain Jitter Correlation", pad=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig_to_image(fig)


# ── Utilities ──────────────────────────────────────────────────────────────

def empty_chart(message: str) -> str:
    return f"<p>{escape(message)}</p>"


def fig_to_image(fig) -> str:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=CHART_DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f'<img alt="chart" src="data:image/png;base64,{encoded}">'
