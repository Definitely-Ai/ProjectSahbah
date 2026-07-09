/**
 * App Controller — orchestrates UI, modes, camera, video, CSV, and results.
 */
import PoseEngine from './pose.js';

(() => {
  'use strict';

  // ── State ─────────────────────────────────────────────────
  let currentMode = 'camera';
  let currentResult = null;
  let csvFile = null;
  let liveJitterHistory = [];  // rolling window of per-frame jitter data
  let prevLandmarks = null;
  let liveChart = null;

  const JITTER_WINDOW = 60;

  // ── DOM refs ──────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  // ── Init ──────────────────────────────────────────────────
  function init() {
    Logger.info('Pose Jitter Lab v2.0 initializing...');

    // Theme
    const saved = localStorage.getItem('pjl-theme');
    if (saved) document.documentElement.setAttribute('data-theme', saved);
    $('themeToggle').addEventListener('click', toggleTheme);

    // Mobile menu
    $('mobileMenu').addEventListener('click', () => $('navLinks').classList.toggle('open'));
    document.querySelectorAll('.nav-link').forEach(a => a.addEventListener('click', () => $('navLinks').classList.remove('open')));

    // Log panel
    $('logToggle').addEventListener('click', () => {
      const lc = $('logContent');
      lc.style.display = lc.style.display === 'none' ? 'block' : 'none';
    });

    // Mode tabs
    document.querySelectorAll('.mode-tab').forEach(tab => {
      tab.addEventListener('click', () => switchMode(tab.dataset.mode));
    });
    positionTabIndicator();

    // Camera controls
    $('cameraStart').addEventListener('click', handleCameraStart);
    $('cameraStop').addEventListener('click', handleCameraStop);
    $('cameraFlip').addEventListener('click', handleCameraFlip);
    $('recordBtn').addEventListener('click', handleRecord);
    $('analyzeRecording').addEventListener('click', handleAnalyzeRecording);
    $('cameraSelect').addEventListener('change', handleCameraChange);

    // Video controls
    setupDropZone($('videoZone'), $('videoInput'), handleVideoFile);
    $('videoCancel').addEventListener('click', handleVideoCancel);
    $('videoAnalyze')?.addEventListener('click', handleVideoAnalyze);

    // CSV controls
    setupDropZone($('csvZone'), $('csvInput'), handleCsvFile);
    $('csvRemove')?.addEventListener('click', clearCsv);
    $('csvAnalyze')?.addEventListener('click', handleCsvAnalyze);

    // Results
    $('exportCsv')?.addEventListener('click', () => currentResult && Analysis.downloadCSV(currentResult.summary, null, 'jitter_summary.csv'));
    $('exportFrames')?.addEventListener('click', () => currentResult && Analysis.downloadCSV(currentResult.jitter, null, 'jitter_frames.csv'));
    $('resetBtn')?.addEventListener('click', resetAll);

    // Error
    $('toastClose')?.addEventListener('click', hideError);

    // GSAP animations
    initAnimations();

    // Auto-load model
    loadPoseModel();

    Logger.info('App initialized.');
  }

  // ── Pose Model Loading ────────────────────────────────────

  async function loadPoseModel() {
    const loader = $('modelLoader');
    const viewport = $('cameraViewport');
    const progressBar = $('modelProgress');

    PoseEngine.onModelProgress = (pct, msg) => {
      if (progressBar) progressBar.style.width = pct + '%';
      Logger.info(`Model: ${msg} (${pct}%)`);
    };

    PoseEngine.onModelLoaded = async () => {
      if (loader) loader.style.display = 'none';
      if (viewport) viewport.style.display = 'block';

      // Enumerate cameras
      const cameras = await PoseEngine.enumerateCameras();
      const select = $('cameraSelect');
      select.innerHTML = '<option value="">Default camera</option>';
      cameras.forEach((cam, i) => {
        const opt = document.createElement('option');
        opt.value = cam.deviceId;
        opt.textContent = cam.label || `Camera ${i + 1}`;
        select.appendChild(opt);
      });
    };

    PoseEngine.onLandmarks = handleLandmarks;
    PoseEngine.onFps = (fps) => {
      const el = $('hudFps');
      if (el) el.textContent = fps;
    };

    try {
      await PoseEngine.loadModel();
    } catch (err) {
      showError('Model Load Failed', err.message);
    }
  }

  // ── Mode Switching ────────────────────────────────────────

  function switchMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
    document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
    const panelMap = { camera: 'panelCamera', video: 'panelVideo', csv: 'panelCsv' };
    $(panelMap[mode])?.classList.add('active');
    positionTabIndicator();
    Logger.info(`Mode: ${mode}`);
  }

  function positionTabIndicator() {
    const active = document.querySelector('.mode-tab.active');
    const indicator = $('tabIndicator');
    if (active && indicator) {
      indicator.style.width = active.offsetWidth + 'px';
      indicator.style.left = active.offsetLeft + 'px';
    }
  }

  // ── Camera Handlers ───────────────────────────────────────

  async function handleCameraStart() {
    const video = $('cameraVideo');
    const canvas = $('cameraCanvas');
    const overlay = $('viewportOverlay');
    const hud = $('liveHud');
    const dash = $('liveDashboard');

    try {
      const deviceId = $('cameraSelect').value || null;
      const info = await PoseEngine.startCamera(video, canvas, deviceId);

      overlay.style.display = 'none';
      hud.style.display = 'flex';
      dash.style.display = 'block';
      $('cameraStart').style.display = 'none';
      $('recordBtn').style.display = 'inline-flex';
      $('cameraStop').style.display = 'inline-flex';

      Logger.info(`Camera started: ${info.width}×${info.height} (${info.label})`);

      // Initialize live chart
      initLiveChart();

    } catch (err) {
      showError('Camera Error', err.message);
    }
  }

  function handleCameraStop() {
    PoseEngine.stopCamera();
    if (PoseEngine.isRecording) PoseEngine.stopRecording();

    $('viewportOverlay').style.display = 'flex';
    $('liveHud').style.display = 'none';
    $('cameraStart').style.display = 'inline-flex';
    $('recordBtn').style.display = 'none';
    $('cameraStop').style.display = 'none';
    $('hudRec').style.display = 'none';

    if (PoseEngine.recordedFrameCount > 0) {
      $('analyzeRecording').style.display = 'inline-flex';
    }

    prevLandmarks = null;
    liveJitterHistory = [];
  }

  async function handleCameraFlip() {
    const facing = PoseEngine.flipCamera();
    if (PoseEngine.isCameraActive) {
      PoseEngine.stopCamera();
      await PoseEngine.startCamera($('cameraVideo'), $('cameraCanvas'));
    }
  }

  function handleCameraChange() {
    if (PoseEngine.isCameraActive) {
      handleCameraStop();
      handleCameraStart();
    }
  }

  function handleRecord() {
    const btn = $('recordBtn');
    const hudRec = $('hudRec');

    if (!PoseEngine.isRecording) {
      PoseEngine.startRecording();
      btn.classList.add('recording');
      btn.innerHTML = '<span class="rec-circle rec-active"></span> Stop Rec';
      hudRec.style.display = 'flex';
    } else {
      PoseEngine.stopRecording();
      btn.classList.remove('recording');
      btn.innerHTML = '<span class="rec-circle"></span> Record';
      hudRec.style.display = 'none';
      if (PoseEngine.recordedFrameCount > 0) {
        $('analyzeRecording').style.display = 'inline-flex';
      }
    }
  }

  async function handleAnalyzeRecording() {
    const frames = PoseEngine.getRecordedFrames();
    if (frames.length < 3) {
      showError('Not Enough Data', 'Record at least 3 frames before analyzing.');
      return;
    }

    showLoading('Converting recorded data...');
    await sleep(50);

    const csvText = PoseEngine.framesToCSV(frames);

    showLoading('Running jitter analysis...');
    await sleep(50);

    const result = Analysis.runAnalysis(csvText, {
      shoulderMode: 'trial_median',
      leftShoulder: 'left_shoulder',
      rightShoulder: 'right_shoulder',
    });

    hideLoading();

    if (result.success) {
      currentResult = result;
      renderResults(result);
      $('results').style.display = 'block';
      $('results').scrollIntoView({ behavior: 'smooth' });
    } else {
      showError('Analysis Error', result.error?.message || 'Unknown error');
    }
  }

  // ── Live Landmarks Handler ────────────────────────────────

  function handleLandmarks(frameData, frame) {
    if (!frameData) {
      $('hudLandmarks').textContent = '0/33';
      return;
    }

    $('hudLandmarks').textContent = `${frameData.visibleCount}/33`;

    if (PoseEngine.isRecording) {
      $('hudFrames').textContent = PoseEngine.recordedFrameCount;
    }

    // Compute real-time jitter
    if (prevLandmarks) {
      const jitters = {};
      let totalJitter = 0;
      let count = 0;

      for (const idx of PoseEngine.BODY_INDICES) {
        const curr = frameData.landmarks[idx];
        const prev = prevLandmarks[idx];
        if (!curr || !prev || curr.visibility < 0.5 || prev.visibility < 0.5) continue;

        const dx = curr.x - prev.x;
        const dy = curr.y - prev.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        const norm = frameData.shoulderWidth > 1e-6 ? dist / frameData.shoulderWidth : NaN;

        if (isFinite(norm)) {
          jitters[curr.name] = norm;
          totalJitter += norm;
          count++;
        }
      }

      const avgJitter = count > 0 ? totalJitter / count : NaN;

      // Update dashboard
      updateDashboard(frameData.shoulderWidth, avgJitter, jitters);

      // Push to history for chart
      liveJitterHistory.push({ frame, avgJitter, shoulderWidth: frameData.shoulderWidth });
      if (liveJitterHistory.length > JITTER_WINDOW * 3) {
        liveJitterHistory = liveJitterHistory.slice(-JITTER_WINDOW * 2);
      }

      updateLiveChart();
    }

    prevLandmarks = frameData.landmarks;
  }

  function updateDashboard(shoulderWidth, avgJitter, jitters) {
    const sw = $('dashShoulder');
    const jv = $('dashJitter');
    const pk = $('dashPeak');
    const st = $('dashStability');
    const swBar = $('dashShoulderBar');
    const jBar = $('dashJitterBar');

    if (isFinite(shoulderWidth)) {
      sw.textContent = shoulderWidth.toFixed(4);
      swBar.style.width = Math.min(100, shoulderWidth * 300) + '%';
    }

    if (isFinite(avgJitter)) {
      jv.textContent = avgJitter.toFixed(5);
      jBar.style.width = Math.min(100, avgJitter * 2000) + '%';
    }

    // Find peak joint
    let peakName = '—';
    let peakVal = 0;
    for (const [name, val] of Object.entries(jitters)) {
      if (val > peakVal) { peakVal = val; peakName = name.replace(/_/g, ' '); }
    }
    pk.textContent = peakName;

    // Stability
    if (liveJitterHistory.length > 10) {
      const recent = liveJitterHistory.slice(-20).map(h => h.shoulderWidth).filter(isFinite);
      if (recent.length > 3) {
        const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
        const std = Math.sqrt(recent.reduce((s, v) => s + (v - mean) ** 2, 0) / (recent.length - 1));
        const cv = std / mean;
        st.textContent = cv < 0.03 ? '● Stable' : cv < 0.08 ? '◐ Moderate' : '○ Variable';
        st.style.color = cv < 0.03 ? 'var(--good)' : cv < 0.08 ? 'var(--warn)' : 'var(--danger)';
      }
    }
  }

  function initLiveChart() {
    if (liveChart) liveChart.destroy();
    const ctx = $('liveChart');
    if (!ctx) return;

    liveChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Avg Normalized Jitter',
          data: [],
          borderColor: '#14b8a6',
          backgroundColor: 'rgba(20,184,166,0.1)',
          fill: true,
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        scales: {
          x: { display: false },
          y: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: 'rgba(255,255,255,0.04)' },
            beginAtZero: true,
          },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function updateLiveChart() {
    if (!liveChart) return;
    const data = liveJitterHistory.slice(-JITTER_WINDOW);
    liveChart.data.labels = data.map((_, i) => i);
    liveChart.data.datasets[0].data = data.map(d => isFinite(d.avgJitter) ? d.avgJitter : null);
    liveChart.update('none');
  }

  // ── Video Handlers ────────────────────────────────────────

  let videoFrames = [];

  function handleVideoFile(file) {
    if (!file.type.startsWith('video/')) {
      showError('Invalid File', 'Please upload a video file (MP4, WebM, MOV).');
      return;
    }

    Logger.info(`Video file: ${file.name} (${formatBytes(file.size)})`);

    const video = $('processingVideo');
    const canvas = $('processingCanvas');
    const processor = $('videoProcessor');
    const zone = $('videoZone');

    video.src = URL.createObjectURL(file);
    video.onloadedmetadata = async () => {
      zone.style.display = 'none';
      processor.style.display = 'block';
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      Logger.info(`Video: ${video.videoWidth}×${video.videoHeight}, ${video.duration.toFixed(1)}s`);

      try {
        videoFrames = await PoseEngine.processVideoFile(video, canvas, (pct, frames, duration) => {
          $('videoProgressFill').style.width = pct + '%';
          $('videoProgressText').textContent = Math.round(pct) + '%';
          $('vpFrames').textContent = frames + ' frames';
          $('vpTime').textContent = formatTime(duration);
        });

        $('videoProgressText').textContent = 'Complete!';
        $('videoAnalyze').style.display = 'inline-flex';
        Logger.info(`Video processed: ${videoFrames.length} frames with landmarks.`);
      } catch (err) {
        showError('Video Processing Error', err.message);
      }
    };
  }

  function handleVideoCancel() {
    $('videoProcessor').style.display = 'none';
    $('videoZone').style.display = 'flex';
    videoFrames = [];
  }

  function handleVideoAnalyze() {
    if (videoFrames.length < 3) {
      showError('Not Enough Data', 'Video must have at least 3 detected frames.');
      return;
    }

    const csvText = PoseEngine.framesToCSV(videoFrames, 'video_upload');
    runCsvAnalysis(csvText);
  }

  // ── CSV Handlers ──────────────────────────────────────────

  function handleCsvFile(file) {
    if (!file.name.match(/\.(csv|txt|tsv)$/i)) {
      showError('Invalid File', 'Please upload a CSV, TSV, or TXT file.');
      return;
    }

    csvFile = file;
    $('csvFileName').textContent = file.name;
    $('csvFileSize').textContent = formatBytes(file.size);
    $('csvZone').style.display = 'none';
    $('csvOptions').style.display = 'block';
  }

  function clearCsv() {
    csvFile = null;
    $('csvZone').style.display = 'flex';
    $('csvOptions').style.display = 'none';
    $('csvInput').value = '';
  }

  async function handleCsvAnalyze() {
    if (!csvFile) return;
    showLoading('Reading CSV...');
    const text = await readFile(csvFile);
    runCsvAnalysis(text);
  }

  async function runCsvAnalysis(csvText) {
    showLoading('Running analysis...');
    await sleep(50);

    const options = {
      shoulderMode: $('shoulderMode')?.value || 'trial_median',
      leftShoulder: $('leftShoulder')?.value || 'left_shoulder',
      rightShoulder: $('rightShoulder')?.value || 'right_shoulder',
    };

    const result = Analysis.runAnalysis(csvText, options);
    hideLoading();

    if (result.success) {
      currentResult = result;
      renderResults(result);
      $('results').style.display = 'block';
      $('results').scrollIntoView({ behavior: 'smooth' });
    } else {
      showError('Analysis Error', result.error?.message || 'Unknown error');
    }
  }

  // ── Results Rendering ─────────────────────────────────────

  function renderResults(result) {
    const s = result.stats;
    $('statsGrid').innerHTML = [
      statCard(s.poseRows, 'Pose Rows'),
      statCard(s.framePairs, 'Frame Pairs'),
      statCard(s.trials, 'Trials'),
      statCard(s.joints, 'Joints'),
      statCard(s.sources, 'Sources'),
      statCard(s.shoulderFrames, 'Shoulder Frames'),
    ].join('');

    $('execContent').innerHTML = result.executive.map(f => `<div class="exec-line">${f}</div>`).join('');

    $('insightsGrid').innerHTML = result.insights.map(i => `
      <div class="insight-card">
        <span class="insight-label">${esc(i.label)}</span>
        <span class="insight-value">${esc(i.value)}</span>
        <span class="insight-note">${esc(i.note)}</span>
      </div>`).join('');

    renderTable($('summaryTableWrap'), result.summary,
      ['source', 'domain', 'trial', 'joint', 'frames', 'normalized_mean', 'normalized_std', 'normalized_p95'], 20);

    renderTable($('qualityTableWrap'), result.quality,
      ['check', 'status', 'result']);

    $('resultsSub').textContent = `${s.poseRows.toLocaleString()} rows · ${s.sources} source(s) · ${result.elapsed}s`;

    Charts.renderAll(result);

    // Animate in
    if (typeof gsap !== 'undefined') {
      gsap.from('#results .stats-row > *', { y: 30, opacity: 0, stagger: 0.05, duration: 0.5, ease: 'power2.out' });
      gsap.from('#results .card', { y: 40, opacity: 0, stagger: 0.08, duration: 0.6, ease: 'power2.out', delay: 0.2 });
    }
  }

  function statCard(value, label) {
    return `<div class="stat-card"><strong>${esc(String(value))}</strong><span>${esc(label)}</span></div>`;
  }

  function renderTable(wrap, data, columns, limit) {
    if (!data || data.length === 0) { wrap.innerHTML = '<p class="no-data">No data available.</p>'; return; }
    const rows = (limit ? data.slice(0, limit) : data);
    const cols = columns || Object.keys(data[0]);
    wrap.innerHTML = `<table class="data-table">
      <thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(row => `<tr>${cols.map(c => {
        const v = row[c];
        if (c === 'status') return `<td><span class="badge badge-${v}">${String(v).toUpperCase()}</span></td>`;
        if (typeof v === 'number') return `<td>${isFinite(v) ? v.toFixed(6) : '—'}</td>`;
        return `<td>${esc(String(v ?? ''))}</td>`;
      }).join('')}</tr>`).join('')}</tbody></table>`;
  }

  // ── Drop Zones ────────────────────────────────────────────

  function setupDropZone(zone, input, handler) {
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', e => { if (e.key === 'Enter') input.click(); });
    input.addEventListener('change', e => { if (e.target.files[0]) handler(e.target.files[0]); });
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => { e.preventDefault(); zone.classList.remove('drag-over'); if (e.dataTransfer.files[0]) handler(e.dataTransfer.files[0]); });
  }

  // ── UI Helpers ────────────────────────────────────────────

  function toggleTheme() {
    const next = document.documentElement.getAttribute('data-theme') === 'light' ? '' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('pjl-theme', next);
    Charts.updateTheme();
  }

  function showError(title, msg) {
    $('toastTitle').textContent = title;
    $('toastMsg').textContent = msg;
    $('errorToast').style.display = 'flex';
    Logger.error(`${title}: ${msg}`);
    if (typeof gsap !== 'undefined') gsap.from('#errorToast', { y: 30, opacity: 0, duration: 0.3 });
  }

  function hideError() { $('errorToast').style.display = 'none'; }
  function showLoading(text) { $('loadingText').textContent = text; $('loadingOverlay').style.display = 'flex'; }
  function hideLoading() { $('loadingOverlay').style.display = 'none'; }

  function resetAll() {
    $('results').style.display = 'none';
    Charts.destroyAll();
    currentResult = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ── Animations ────────────────────────────────────────────

  function initAnimations() {
    if (typeof gsap === 'undefined') return;

    gsap.registerPlugin(ScrollTrigger);

    // Hero entrance
    gsap.from('.hero-content', { y: 60, opacity: 0, duration: 0.8, ease: 'power3.out' });
    gsap.from('.hero-badge', { scale: 0.8, opacity: 0, duration: 0.5, delay: 0.2 });
    gsap.from('.hero h1', { y: 40, opacity: 0, duration: 0.7, delay: 0.3 });
    gsap.from('.hero-sub', { y: 30, opacity: 0, duration: 0.6, delay: 0.4 });
    gsap.from('.hero-actions', { y: 30, opacity: 0, duration: 0.6, delay: 0.5 });
    gsap.from('.hero-feat', { y: 20, opacity: 0, stagger: 0.08, duration: 0.5, delay: 0.6 });

    // Scroll-triggered animations
    document.querySelectorAll('.anim-up').forEach(el => {
      gsap.from(el, {
        scrollTrigger: { trigger: el, start: 'top 85%', toggleActions: 'play none none none' },
        y: 40, opacity: 0, duration: 0.6, ease: 'power2.out',
      });
    });

    // Methodology steps
    gsap.from('.method-step', {
      scrollTrigger: { trigger: '.method-steps', start: 'top 80%' },
      y: 50, opacity: 0, stagger: 0.12, duration: 0.7, ease: 'power2.out',
    });
  }

  // ── Utilities ─────────────────────────────────────────────

  function readFile(file) { return new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result); r.onerror = () => rej(new Error('File read failed')); r.readAsText(file); }); }
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  function formatBytes(b) { if (!b) return '0 B'; const k = 1024; const s = ['B', 'KB', 'MB', 'GB']; const i = Math.floor(Math.log(b) / Math.log(k)); return (b / k ** i).toFixed(1) + ' ' + s[i]; }
  function formatTime(s) { const m = Math.floor(s / 60); return m + ':' + String(Math.floor(s % 60)).padStart(2, '0'); }
  function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

  // ── Global error handlers ─────────────────────────────────
  window.addEventListener('error', e => Logger.error(`Uncaught: ${e.message} at ${e.filename}:${e.lineno}`));
  window.addEventListener('unhandledrejection', e => Logger.error(`Promise: ${e.reason}`));

  // ── Boot ──────────────────────────────────────────────────
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', init) : init();
})();
