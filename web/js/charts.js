/**
 * Charts — Chart.js-powered visualizations for jitter analysis.
 *
 * All chart functions accept analysis results and render to canvas elements.
 * Charts are responsive and adapt to light/dark themes.
 */

const Charts = (() => {
  'use strict';

  const instances = {};

  // Color palette
  const TEAL = '#14b8a6';
  const TEAL_50 = 'rgba(20, 184, 166, 0.5)';
  const TEAL_20 = 'rgba(20, 184, 166, 0.2)';
  const ORANGE = '#f97316';
  const ORANGE_50 = 'rgba(249, 115, 22, 0.5)';
  const ORANGE_20 = 'rgba(249, 115, 22, 0.2)';
  const PURPLE = '#a78bfa';
  const PURPLE_50 = 'rgba(167, 139, 250, 0.5)';
  const PURPLE_20 = 'rgba(167, 139, 250, 0.2)';

  const SOURCE_COLORS = [
    { bg: TEAL, fill: TEAL_20, border: TEAL },
    { bg: ORANGE, fill: ORANGE_20, border: ORANGE },
    { bg: PURPLE, fill: PURPLE_20, border: PURPLE },
    { bg: '#f43f5e', fill: 'rgba(244,63,94,0.2)', border: '#f43f5e' },
    { bg: '#06b6d4', fill: 'rgba(6,182,212,0.2)', border: '#06b6d4' },
  ];

  function getSourceColor(index) {
    return SOURCE_COLORS[index % SOURCE_COLORS.length];
  }

  function isDark() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
  }

  function gridColor() { return isDark() ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)'; }
  function textColor() { return isDark() ? '#94a3b8' : '#64748b'; }

  function destroyChart(id) {
    if (instances[id]) {
      instances[id].destroy();
      delete instances[id];
    }
  }

  function baseOptions(title) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 600, easing: 'easeOutQuart' },
      plugins: {
        legend: {
          labels: { color: textColor(), font: { family: "'Inter', sans-serif", size: 11, weight: 600 } },
        },
        title: title ? {
          display: true,
          text: title,
          color: textColor(),
          font: { family: "'Inter', sans-serif", size: 14, weight: 700 },
          padding: { bottom: 12 },
        } : { display: false },
        tooltip: {
          backgroundColor: isDark() ? '#1e293b' : '#ffffff',
          titleColor: isDark() ? '#e2e8f0' : '#0f172a',
          bodyColor: isDark() ? '#cbd5e1' : '#334155',
          borderColor: isDark() ? '#334155' : '#e2e8f0',
          borderWidth: 1,
          cornerRadius: 8,
          padding: 10,
          titleFont: { family: "'Inter', sans-serif", weight: 700 },
          bodyFont: { family: "'Inter', sans-serif" },
        },
      },
      scales: {
        x: { ticks: { color: textColor() }, grid: { color: gridColor() } },
        y: { ticks: { color: textColor() }, grid: { color: gridColor() } },
      },
    };
  }


  // ── Bar chart: Normalized jitter by joint ──────────────────

  function renderBarChart(canvasId, summary) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx || summary.length === 0) return;

    // Group by source, show top 16 joints
    const sources = [...new Set(summary.map(s => s.source))];
    const joints = [...new Set(summary.map(s => s.joint))]
      .filter(j => !j.includes('shoulder'));

    // Sort joints by max mean jitter
    joints.sort((a, b) => {
      const aMax = Math.max(...summary.filter(s => s.joint === a).map(s => s.normalized_mean));
      const bMax = Math.max(...summary.filter(s => s.joint === b).map(s => s.normalized_mean));
      return bMax - aMax;
    });
    const topJoints = joints.slice(0, 16);

    const datasets = sources.map((source, i) => {
      const color = getSourceColor(i);
      return {
        label: source,
        data: topJoints.map(joint => {
          const row = summary.find(s => s.source === source && s.joint === joint);
          return row ? row.normalized_mean : 0;
        }),
        backgroundColor: color.bg + 'cc',
        borderColor: color.border,
        borderWidth: 1,
        borderRadius: 4,
      };
    });

    const opts = baseOptions('');
    opts.indexAxis = 'y';
    opts.scales.x.title = { display: true, text: 'Mean Normalized Jitter (shoulder widths/frame)', color: textColor() };
    opts.plugins.legend.position = 'top';

    instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: { labels: topJoints.map(j => j.replace(/_/g, ' ')), datasets },
      options: opts,
    });
  }


  // ── Box plot (violin-like): Jitter distribution ───────────

  function renderDistribution(canvasId, jitter) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx || jitter.length === 0) return;

    const groups = {};
    jitter.forEach(j => {
      const key = `${j.source} | ${j.domain}`;
      if (!groups[key]) groups[key] = [];
      if (isFinite(j.normalized_jitter)) groups[key].push(j.normalized_jitter);
    });

    const labels = Object.keys(groups);
    const datasets = [];

    labels.forEach((label, i) => {
      const vals = groups[label].sort((a, b) => a - b);
      const q1 = percentile(vals, 25);
      const med = percentile(vals, 50);
      const q3 = percentile(vals, 75);
      const iqr = q3 - q1;
      const whiskerLow = Math.max(vals[0], q1 - 1.5 * iqr);
      const whiskerHigh = Math.min(vals[vals.length - 1], q3 + 1.5 * iqr);

      // Render as a floating bar (Q1 to Q3) + error bars
      const color = getSourceColor(i);
      datasets.push({
        label: label,
        data: [{ x: i, y: [q1, q3] }],
        backgroundColor: color.bg + '88',
        borderColor: color.border,
        borderWidth: 2,
        _stats: { q1, med, q3, whiskerLow, whiskerHigh, n: vals.length },
      });
    });

    // Use a simpler bar chart with annotations since Chart.js doesn't have native box plots
    const color = i => getSourceColor(i);
    const barData = labels.map((label, i) => {
      const vals = groups[label];
      return {
        mean: vals.reduce((a, b) => a + b, 0) / vals.length,
        std: Math.sqrt(vals.reduce((s, v) => s + (v - vals.reduce((a, b) => a + b, 0) / vals.length) ** 2, 0) / (vals.length - 1)),
        p5: percentile(vals.sort((a, b) => a - b), 5),
        p95: percentile(vals, 95),
        median: percentile(vals, 50),
      };
    });

    const opts = baseOptions('');
    opts.scales.y.title = { display: true, text: 'Normalized Jitter', color: textColor() };

    instances[canvasId] = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels.map(l => l.replace(/_/g, ' ')),
        datasets: [{
          label: 'Mean ± Std',
          data: barData.map(d => d.mean),
          backgroundColor: labels.map((_, i) => getSourceColor(i).bg + '88'),
          borderColor: labels.map((_, i) => getSourceColor(i).border),
          borderWidth: 2,
          borderRadius: 6,
          errorBars: barData,
        }],
      },
      options: opts,
    });
  }


  // ── Temporal trace ────────────────────────────────────────

  function renderTemporal(canvasId, jitter) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx || jitter.length === 0) return;

    // Find top 3 joints by mean jitter (excluding shoulders)
    const jointMeans = {};
    jitter.forEach(j => {
      if (j.joint.includes('shoulder') || !isFinite(j.normalized_jitter)) return;
      if (!jointMeans[j.joint]) jointMeans[j.joint] = [];
      jointMeans[j.joint].push(j.normalized_jitter);
    });
    const topJoints = Object.entries(jointMeans)
      .map(([joint, vals]) => ({ joint, mean: vals.reduce((a, b) => a + b, 0) / vals.length }))
      .sort((a, b) => b.mean - a.mean)
      .slice(0, 3)
      .map(j => j.joint);

    if (topJoints.length === 0) return;

    const datasets = [];
    let colorIdx = 0;

    for (const joint of topJoints) {
      const sources = [...new Set(jitter.filter(j => j.joint === joint).map(j => j.source))];
      for (const source of sources) {
        const rows = jitter
          .filter(j => j.joint === joint && j.source === source && isFinite(j.normalized_jitter))
          .sort((a, b) => a.frame_to - b.frame_to);

        if (rows.length < 2) continue;
        const color = getSourceColor(colorIdx++);

        datasets.push({
          label: `${joint.replace(/_/g, ' ')} (${source})`,
          data: rows.map(r => ({ x: r.frame_to, y: r.normalized_jitter })),
          borderColor: color.border,
          backgroundColor: color.fill,
          fill: false,
          borderWidth: 1.2,
          pointRadius: 0,
          pointHoverRadius: 3,
          tension: 0.1,
        });
      }
    }

    const opts = baseOptions('');
    opts.scales.x.type = 'linear';
    opts.scales.x.title = { display: true, text: 'Frame', color: textColor() };
    opts.scales.y.title = { display: true, text: 'Normalized Jitter', color: textColor() };
    opts.plugins.legend.position = 'top';

    instances[canvasId] = new Chart(ctx, { type: 'line', data: { datasets }, options: opts });
  }


  // ── CDF overlay ───────────────────────────────────────────

  function renderCDF(canvasId, jitter) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx || jitter.length === 0) return;

    const groups = {};
    jitter.forEach(j => {
      const key = `${j.source} | ${j.domain}`;
      if (!groups[key]) groups[key] = [];
      if (isFinite(j.normalized_jitter)) groups[key].push(j.normalized_jitter);
    });

    const datasets = [];
    let colorIdx = 0;
    for (const [label, vals] of Object.entries(groups)) {
      if (vals.length < 5) continue;
      vals.sort((a, b) => a - b);
      const color = getSourceColor(colorIdx++);

      // Downsample for performance if > 2000 points
      let plotVals = vals;
      if (vals.length > 2000) {
        const step = Math.ceil(vals.length / 2000);
        plotVals = vals.filter((_, i) => i % step === 0);
      }

      datasets.push({
        label: label.replace(/_/g, ' '),
        data: plotVals.map((v, i) => ({ x: v, y: (i + 1) / plotVals.length })),
        borderColor: color.border,
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 3,
        stepped: 'after',
      });
    }

    const opts = baseOptions('');
    opts.scales.x.type = 'linear';
    opts.scales.x.title = { display: true, text: 'Normalized Jitter', color: textColor() };
    opts.scales.y.title = { display: true, text: 'Cumulative Proportion', color: textColor() };
    opts.scales.y.max = 1.02;

    instances[canvasId] = new Chart(ctx, { type: 'line', data: { datasets }, options: opts });
  }


  // ── Shoulder width stability ──────────────────────────────

  function renderShoulder(canvasId, shoulderWidths) {
    destroyChart(canvasId);
    const ctx = document.getElementById(canvasId);
    if (!ctx || shoulderWidths.length === 0) return;

    const groups = {};
    shoulderWidths.forEach(w => {
      const key = `${w.source} | ${w.domain} | ${w.trial}`;
      if (!groups[key]) groups[key] = [];
      if (isFinite(w.shoulder_width)) groups[key].push(w);
    });

    const datasets = [];
    let colorIdx = 0;
    for (const [label, rows] of Object.entries(groups)) {
      rows.sort((a, b) => a.frame - b.frame);
      const color = getSourceColor(colorIdx++);
      datasets.push({
        label: label.replace(/_/g, ' '),
        data: rows.map(r => ({ x: r.frame, y: r.shoulder_width })),
        borderColor: color.border,
        backgroundColor: color.fill,
        fill: false,
        borderWidth: 1.8,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.2,
      });
    }

    const opts = baseOptions('');
    opts.scales.x.type = 'linear';
    opts.scales.x.title = { display: true, text: 'Frame', color: textColor() };
    opts.scales.y.title = { display: true, text: 'Shoulder Width (domain units)', color: textColor() };

    instances[canvasId] = new Chart(ctx, { type: 'line', data: { datasets }, options: opts });
  }


  // ── Render all charts ─────────────────────────────────────

  function renderAll(result) {
    Logger.info('Rendering charts...');
    try {
      renderBarChart('chartBar', result.summary);
      renderDistribution('chartBox', result.jitter);
      renderTemporal('chartTemporal', result.jitter);
      renderCDF('chartCdf', result.jitter);
      renderShoulder('chartShoulder', result.shoulderWidths);
      Logger.info('Charts rendered successfully.');
    } catch (e) {
      Logger.error(`Chart rendering error: ${e.message}`);
    }
  }

  function destroyAll() {
    Object.keys(instances).forEach(destroyChart);
  }

  // Theme update
  function updateTheme() {
    // Re-render all existing charts with new colors
    Object.values(instances).forEach(chart => {
      if (chart.options) {
        chart.options.scales.x.ticks.color = textColor();
        chart.options.scales.y.ticks.color = textColor();
        chart.options.scales.x.grid.color = gridColor();
        chart.options.scales.y.grid.color = gridColor();
        if (chart.options.plugins.legend) {
          chart.options.plugins.legend.labels.color = textColor();
        }
        chart.update('none');
      }
    });
  }


  // ── Utilities ─────────────────────────────────────────────

  function percentile(sortedArr, p) {
    if (sortedArr.length === 0) return 0;
    const idx = (p / 100) * (sortedArr.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sortedArr[lo];
    return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
  }


  return { renderAll, destroyAll, updateTheme };
})();

window.Charts = Charts;
