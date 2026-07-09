/**
 * Analysis Engine — client-side jitter computation.
 *
 * Ports the core Python metrics to JavaScript so the entire analysis
 * runs in the browser with zero server calls.
 */

const Analysis = (() => {
  'use strict';

  // ── CSV Parsing ───────────────────────────────────────────

  /**
   * Parse a pose CSV string into structured row objects.
   * Auto-detects column aliases (x_norm, X_Norm, x_mm, etc.).
   */
  function parseCSV(text) {
    Logger.info('Parsing CSV...');
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) throw new AnalysisError('CSV must have a header row and at least one data row.', 'PARSE_ERROR');

    const rawHeaders = lines[0].split(',').map(h => h.trim());
    Logger.debug('CSV headers:', rawHeaders.join(', '));

    // Build column alias map
    const aliasMap = buildAliasMap(rawHeaders);
    Logger.info(`Detected columns: ${Object.keys(aliasMap).filter(k => aliasMap[k] !== null).join(', ')}`);

    // Validate required columns
    if (aliasMap.joint === null) throw new AnalysisError('CSV must contain a "joint" or "landmark" column.', 'MISSING_COLUMN');
    if (aliasMap.frame === null) throw new AnalysisError('CSV must contain a "frame" column.', 'MISSING_COLUMN');

    const rows = [];
    let parseErrors = 0;
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      try {
        const values = lines[i].split(',').map(v => v.trim());
        const row = {};
        for (const [key, idx] of Object.entries(aliasMap)) {
          if (idx === null) continue;
          const val = values[idx];
          if (['frame', 'x', 'y', 'z', 'x_mm', 'y_mm', 'z_mm'].includes(key)) {
            row[key] = val === '' || val === undefined ? NaN : parseFloat(val);
          } else {
            row[key] = val || '';
          }
        }
        rows.push(row);
      } catch (e) {
        parseErrors++;
        if (parseErrors <= 5) Logger.warn(`Parse error on line ${i + 1}: ${e.message}`);
      }
    }

    if (parseErrors > 5) Logger.warn(`...and ${parseErrors - 5} more parse errors.`);
    if (rows.length === 0) throw new AnalysisError('No valid data rows found in CSV.', 'EMPTY_DATA');

    Logger.info(`Parsed ${rows.length} rows from ${lines.length - 1} lines (${parseErrors} errors).`);
    return rows;
  }

  function buildAliasMap(headers) {
    const lower = headers.map(h => h.toLowerCase().replace(/[\s-]/g, '_'));
    const find = (...aliases) => {
      for (const alias of aliases) {
        const idx = lower.indexOf(alias.toLowerCase().replace(/[\s-]/g, '_'));
        if (idx >= 0) return idx;
      }
      return null;
    };

    return {
      source: find('source', 'system', 'camera'),
      domain: find('domain', 'coord_domain', 'coordinate_domain'),
      trial: find('trial', 'trial_name', 'session'),
      frame: find('frame', 'frame_number', 'frame_id', 'frame_index'),
      joint: find('joint', 'landmark', 'keypoint', 'joint_name', 'landmark_name'),
      x: find('x', 'x_norm', 'x_normalized', 'x_2d', 'norm_x'),
      y: find('y', 'y_norm', 'y_normalized', 'y_2d', 'norm_y'),
      z: find('z', 'z_norm', 'z_normalized', 'z_2d', 'norm_z'),
      x_mm: find('x_mm', 'x_3d', 'x_stereo', 'x_world'),
      y_mm: find('y_mm', 'y_3d', 'y_stereo', 'y_world'),
      z_mm: find('z_mm', 'z_3d', 'z_stereo', 'z_world'),
      timestamp: find('timestamp', 'time', 'time_s', 'time_sec'),
    };
  }


  // ── Domain Detection ──────────────────────────────────────

  function detectDomains(rows) {
    const hasNorm = rows.some(r => isFinite(r.x) && isFinite(r.y));
    const hasMM = rows.some(r => isFinite(r.x_mm) && isFinite(r.y_mm));
    const domains = [];
    if (hasNorm) domains.push({ name: 'normalized_2d', xKey: 'x', yKey: 'y', zKey: 'z' });
    if (hasMM) domains.push({ name: 'mm_3d', xKey: 'x_mm', yKey: 'y_mm', zKey: 'z_mm' });
    if (domains.length === 0) {
      // Fallback: use whatever x/y columns exist
      domains.push({ name: 'default', xKey: 'x', yKey: 'y', zKey: 'z' });
    }
    Logger.info(`Detected domains: ${domains.map(d => d.name).join(', ')}`);
    return domains;
  }


  // ── Shoulder Width ────────────────────────────────────────

  function computeShoulderWidths(rows, domain, leftName, rightName) {
    Logger.info(`Computing shoulder widths (${domain.name})...`);
    const frames = groupBy(rows, 'frame');
    const widths = [];

    for (const [frameStr, frameRows] of Object.entries(frames)) {
      const frame = parseInt(frameStr);
      const left = frameRows.find(r => r.joint === leftName);
      const right = frameRows.find(r => r.joint === rightName);
      if (!left || !right) continue;

      const lx = left[domain.xKey], ly = left[domain.yKey], lz = left[domain.zKey];
      const rx = right[domain.xKey], ry = right[domain.yKey], rz = right[domain.zKey];

      if (!isFinite(lx) || !isFinite(ly) || !isFinite(rx) || !isFinite(ry)) continue;

      let dist;
      if (isFinite(lz) && isFinite(rz)) {
        dist = Math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2 + (rz - lz) ** 2);
      } else {
        dist = Math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2);
      }

      const source = left.source || 'default';
      const trial = left.trial || 'trial_1';
      widths.push({ source, domain: domain.name, trial, frame, shoulder_width: dist });
    }

    Logger.info(`Computed ${widths.length} shoulder width measurements.`);
    return widths;
  }

  function medianShoulderWidth(shoulderWidths, source, domain, trial) {
    const vals = shoulderWidths
      .filter(w => w.source === source && w.domain === domain && w.trial === trial)
      .map(w => w.shoulder_width)
      .filter(isFinite)
      .sort((a, b) => a - b);
    if (vals.length === 0) return NaN;
    const mid = Math.floor(vals.length / 2);
    return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2;
  }


  // ── Jitter Computation ────────────────────────────────────

  function computeJitter(rows, domain, shoulderWidths, shoulderMode, leftName, rightName) {
    Logger.info(`Computing jitter (${domain.name}, mode=${shoulderMode})...`);
    const jitter = [];
    const joints = [...new Set(rows.map(r => r.joint))];
    const sources = [...new Set(rows.map(r => r.source || 'default'))];
    const trials = [...new Set(rows.map(r => r.trial || 'trial_1'))];

    for (const source of sources) {
      for (const trial of trials) {
        const medianSW = medianShoulderWidth(shoulderWidths, source, domain.name, trial);
        if (!isFinite(medianSW) || medianSW < 1e-12) {
          Logger.warn(`No valid shoulder width for ${source}/${domain.name}/${trial}, skipping normalization.`);
        }

        for (const joint of joints) {
          const jointRows = rows
            .filter(r => r.joint === joint && (r.source || 'default') === source && (r.trial || 'trial_1') === trial)
            .sort((a, b) => a.frame - b.frame);

          for (let i = 1; i < jointRows.length; i++) {
            const prev = jointRows[i - 1];
            const curr = jointRows[i];

            const px = prev[domain.xKey], py = prev[domain.yKey], pz = prev[domain.zKey];
            const cx = curr[domain.xKey], cy = curr[domain.yKey], cz = curr[domain.zKey];

            if (!isFinite(px) || !isFinite(py) || !isFinite(cx) || !isFinite(cy)) continue;

            let rawDist;
            if (isFinite(pz) && isFinite(cz)) {
              rawDist = Math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2);
            } else {
              rawDist = Math.sqrt((cx - px) ** 2 + (cy - py) ** 2);
            }

            const frameGap = curr.frame - prev.frame;
            const scale = (shoulderMode === 'trial_median') ? medianSW : medianSW;
            const normalized = (isFinite(scale) && scale > 1e-12) ? rawDist / scale : NaN;

            jitter.push({
              source,
              domain: domain.name,
              trial,
              joint,
              frame_from: prev.frame,
              frame_to: curr.frame,
              frame_gap: frameGap,
              raw_jitter: rawDist,
              shoulder_scale: scale,
              normalized_jitter: normalized,
              scale_valid: isFinite(scale) && scale > 1e-12,
            });
          }
        }
      }
    }

    Logger.info(`Computed ${jitter.length} jitter rows.`);
    return jitter;
  }


  // ── Summary ───────────────────────────────────────────────

  function summarizeJitter(jitter) {
    Logger.info('Summarizing jitter...');
    const summary = [];
    const groups = groupByMulti(jitter, ['source', 'domain', 'trial', 'joint']);

    for (const [key, rows] of Object.entries(groups)) {
      const vals = rows.map(r => r.normalized_jitter).filter(isFinite);
      if (vals.length === 0) continue;

      vals.sort((a, b) => a - b);
      const rawVals = rows.map(r => r.raw_jitter).filter(isFinite);

      const [source, domain, trial, joint] = key.split('|||');
      summary.push({
        source, domain, trial, joint,
        frames: vals.length,
        raw_mean: mean(rawVals),
        raw_std: std(rawVals),
        normalized_mean: mean(vals),
        normalized_std: std(vals),
        normalized_median: median(vals),
        normalized_p95: percentile(vals, 95),
        normalized_max: Math.max(...vals),
        normalized_min: Math.min(...vals),
      });
    }

    summary.sort((a, b) => b.normalized_mean - a.normalized_mean);
    Logger.info(`Summary: ${summary.length} joint-trial combinations.`);
    return summary;
  }


  // ── Quality Gates ─────────────────────────────────────────

  function qualityGates(jitter, shoulderWidths) {
    const totalRows = jitter.length;
    const validRows = jitter.filter(r => isFinite(r.normalized_jitter)).length;
    const missingShoulder = shoulderWidths.filter(w => !isFinite(w.shoulder_width)).length;

    // Max CV across all source/trial combos
    let maxCV = NaN;
    const swGroups = groupByMulti(shoulderWidths, ['source', 'domain', 'trial']);
    for (const rows of Object.values(swGroups)) {
      const vals = rows.map(r => r.shoulder_width).filter(isFinite);
      if (vals.length < 3) continue;
      const cv = std(vals) / mean(vals);
      if (!isFinite(maxCV) || cv > maxCV) maxCV = cv;
    }

    return [
      {
        check: 'Normalized jitter denominator',
        status: totalRows > 0 && validRows === totalRows ? 'pass' : 'review',
        result: `${validRows} of ${totalRows} rows valid`,
      },
      {
        check: 'Shoulder frames',
        status: missingShoulder === 0 && shoulderWidths.length > 0 ? 'pass' : 'review',
        result: `${missingShoulder} missing shoulder-width frames`,
      },
      {
        check: 'Shoulder-width stability',
        status: isFinite(maxCV) && maxCV < 0.05 ? 'pass' : 'review',
        result: isFinite(maxCV) ? `max CV ${(maxCV * 100).toFixed(2)}%` : 'not available',
      },
    ];
  }


  // ── Executive Summary ─────────────────────────────────────

  function executiveSummary(summary, jitter) {
    const findings = [];

    const sources = [...new Set(summary.map(s => s.source))];
    if (sources.length >= 2) {
      const sourceMeans = {};
      for (const source of sources) {
        const vals = summary.filter(s => s.source === source).map(s => s.normalized_mean).filter(isFinite);
        sourceMeans[source] = mean(vals);
      }
      const sorted = Object.entries(sourceMeans).sort((a, b) => b[1] - a[1]);
      const [hiSrc, hiVal] = sorted[0];
      const [loSrc, loVal] = sorted[sorted.length - 1];
      const ratio = loVal > 1e-12 ? hiVal / loVal : Infinity;
      findings.push(
        `<strong>${hiSrc}</strong> exhibited ${ratio.toFixed(1)}× higher mean normalized jitter ` +
        `than <strong>${loSrc}</strong> (${hiVal.toFixed(4)} vs ${loVal.toFixed(4)} shoulder widths/frame).`
      );
    }

    if (summary.length > 0) {
      const top = summary[0]; // Already sorted descending
      findings.push(
        `The highest per-joint jitter was observed at <strong>${top.joint}</strong> ` +
        `(${top.source}) with a mean of ${top.normalized_mean.toFixed(4)} shoulder widths/frame.`
      );
    }

    const validRate = jitter.length > 0
      ? (jitter.filter(r => isFinite(r.normalized_jitter)).length / jitter.length * 100)
      : 0;
    findings.push(`Data integrity: ${validRate.toFixed(1)}% of jitter rows have valid normalization.`);

    return findings;
  }


  // ── Insight Cards ─────────────────────────────────────────

  function insightCards(summary, jitter, shoulderWidths) {
    const validRate = jitter.length > 0
      ? (jitter.filter(r => isFinite(r.normalized_jitter)).length / jitter.length * 100).toFixed(1)
      : '0.0';

    let peak = { value: 'No valid jitter', note: 'Add at least two frames per joint.' };
    if (summary.length > 0) {
      const top = summary[0];
      peak = {
        value: `${top.joint} | ${top.source}`,
        note: `Mean normalized jitter ${top.normalized_mean.toFixed(4)} shoulder widths/frame.`,
      };
    }

    let gap = { value: 'Not available', note: 'Need at least two sources.' };
    const sources = [...new Set(summary.map(s => s.source))];
    if (sources.length >= 2) {
      let bestRatio = 0, bestJoint = '', bestHi = '', bestLo = '';
      for (const joint of [...new Set(summary.map(s => s.joint))]) {
        if (joint.includes('shoulder')) continue;
        const jointData = summary.filter(s => s.joint === joint);
        if (jointData.length < 2) continue;
        const sorted = jointData.sort((a, b) => b.normalized_mean - a.normalized_mean);
        const ratio = sorted[sorted.length - 1].normalized_mean > 1e-12
          ? sorted[0].normalized_mean / sorted[sorted.length - 1].normalized_mean : 0;
        if (ratio > bestRatio) {
          bestRatio = ratio;
          bestJoint = joint;
          bestHi = sorted[0].source;
          bestLo = sorted[sorted.length - 1].source;
        }
      }
      if (bestRatio > 0) {
        gap = {
          value: `${bestRatio.toFixed(2)}x on ${bestJoint}`,
          note: `${bestHi} exceeded ${bestLo}.`,
        };
      }
    }

    let stability = { value: 'No shoulders', note: 'Shoulder landmarks required.' };
    if (shoulderWidths.length > 0) {
      const swGroups = groupByMulti(shoulderWidths, ['source', 'domain', 'trial']);
      let worstCV = 0, worstKey = '';
      for (const [key, rows] of Object.entries(swGroups)) {
        const vals = rows.map(r => r.shoulder_width).filter(isFinite);
        if (vals.length < 3) continue;
        const cv = std(vals) / mean(vals);
        if (cv > worstCV) { worstCV = cv; worstKey = key.replace(/\|\|\|/g, ' | '); }
      }
      if (worstCV > 0) {
        stability = {
          value: `${(worstCV * 100).toFixed(2)}% max CV`,
          note: `Worst: ${worstKey}`,
        };
      }
    }

    return [
      { label: 'DATA INTEGRITY', value: `${validRate}% valid`, note: 'Normalized jitter rows with a valid shoulder denominator.' },
      { label: 'PEAK FINDING', value: peak.value, note: peak.note },
      { label: 'CROSS-DOMAIN GAP', value: gap.value, note: gap.note },
      { label: 'SHOULDER STABILITY', value: stability.value, note: stability.note },
    ];
  }


  // ── Full Pipeline ─────────────────────────────────────────

  function runAnalysis(csvText, options = {}) {
    const startTime = performance.now();
    Logger.info('=== Starting analysis pipeline ===');

    try {
      const leftShoulder = options.leftShoulder || 'left_shoulder';
      const rightShoulder = options.rightShoulder || 'right_shoulder';
      const shoulderMode = options.shoulderMode || 'trial_median';

      // 1. Parse CSV
      const rows = parseCSV(csvText);

      // 2. Fill defaults
      rows.forEach(r => {
        if (!r.source) r.source = 'default';
        if (!r.trial) r.trial = 'trial_1';
      });

      // 3. Detect domains
      const domains = detectDomains(rows);

      // 4. Compute per domain
      let allJitter = [];
      let allShoulderWidths = [];
      let allSummary = [];

      for (const domain of domains) {
        const sw = computeShoulderWidths(rows, domain, leftShoulder, rightShoulder);
        allShoulderWidths = allShoulderWidths.concat(sw);

        const jitter = computeJitter(rows, domain, sw, shoulderMode, leftShoulder, rightShoulder);
        allJitter = allJitter.concat(jitter);

        const summary = summarizeJitter(jitter);
        allSummary = allSummary.concat(summary);
      }

      // Sort summary descending by normalized_mean
      allSummary.sort((a, b) => b.normalized_mean - a.normalized_mean);

      // 5. Quality
      const quality = qualityGates(allJitter, allShoulderWidths);

      // 6. Executive summary
      const exec = executiveSummary(allSummary, allJitter);

      // 7. Insights
      const insights = insightCards(allSummary, allJitter, allShoulderWidths);

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      Logger.info(`=== Analysis complete in ${elapsed}s ===`);
      Logger.info(`  Rows: ${rows.length}, Jitter: ${allJitter.length}, Summary: ${allSummary.length}`);

      return {
        success: true,
        rows,
        jitter: allJitter,
        summary: allSummary,
        shoulderWidths: allShoulderWidths,
        quality,
        executive: exec,
        insights,
        domains,
        stats: {
          poseRows: rows.length,
          framePairs: allJitter.length,
          trials: [...new Set(rows.map(r => r.trial))].length,
          joints: [...new Set(rows.map(r => r.joint))].length,
          sources: [...new Set(rows.map(r => r.source))].length,
          shoulderFrames: allShoulderWidths.filter(w => isFinite(w.shoulder_width)).length,
        },
        elapsed,
      };
    } catch (e) {
      Logger.error(`Analysis failed: ${e.message}`);
      return { success: false, error: e };
    }
  }


  // ── CSV Export ─────────────────────────────────────────────

  function toCSV(data, columns) {
    if (!data || data.length === 0) return '';
    const cols = columns || Object.keys(data[0]);
    const header = cols.join(',');
    const rows = data.map(row => cols.map(c => {
      const v = row[c];
      if (v === undefined || v === null) return '';
      if (typeof v === 'string' && v.includes(',')) return `"${v}"`;
      if (typeof v === 'number') return isFinite(v) ? v.toString() : '';
      return String(v);
    }).join(','));
    return [header, ...rows].join('\n');
  }

  function downloadCSV(data, columns, filename) {
    const csv = toCSV(data, columns);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    Logger.info(`Downloaded ${filename} (${data.length} rows).`);
  }


  // ── Utilities ─────────────────────────────────────────────

  function groupBy(arr, key) {
    const groups = {};
    for (const item of arr) {
      const k = String(item[key] ?? '');
      if (!groups[k]) groups[k] = [];
      groups[k].push(item);
    }
    return groups;
  }

  function groupByMulti(arr, keys) {
    const groups = {};
    for (const item of arr) {
      const k = keys.map(key => String(item[key] ?? '')).join('|||');
      if (!groups[k]) groups[k] = [];
      groups[k].push(item);
    }
    return groups;
  }

  function mean(arr) {
    if (arr.length === 0) return NaN;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }

  function std(arr) {
    if (arr.length < 2) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
  }

  function median(arr) {
    if (arr.length === 0) return NaN;
    const sorted = [...arr].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }

  function percentile(sortedArr, p) {
    if (sortedArr.length === 0) return NaN;
    const idx = (p / 100) * (sortedArr.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sortedArr[lo];
    return sortedArr[lo] + (sortedArr[hi] - sortedArr[lo]) * (idx - lo);
  }


  // ── Custom Error ──────────────────────────────────────────

  class AnalysisError extends Error {
    constructor(message, code) {
      super(message);
      this.name = 'AnalysisError';
      this.code = code;
    }
  }

  // Public API
  return {
    runAnalysis,
    parseCSV,
    toCSV,
    downloadCSV,
    AnalysisError,
  };
})();

window.Analysis = Analysis;
