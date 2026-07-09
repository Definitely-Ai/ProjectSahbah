/**
 * Pose Detection Module — MediaPipe Pose Landmarker integration.
 *
 * Handles camera access, video file processing, landmark detection,
 * skeleton drawing, and real-time jitter computation.
 * Requests highest available resolution for accurate landmark placement.
 */

import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';

// ── Constants ──────────────────────────────────────────────
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const LANDMARK_NAMES = [
  'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
  'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
  'left_index', 'right_index', 'left_thumb', 'right_thumb',
  'left_hip', 'right_hip', 'left_knee', 'right_knee',
  'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
  'left_foot_index', 'right_foot_index',
];

// Body-relevant landmarks (skip face details for jitter analysis)
const BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const LEFT_SHOULDER_IDX = 11;
const RIGHT_SHOULDER_IDX = 12;
const VISIBILITY_THRESHOLD = 0.5;

// High-resolution camera constraints for accurate landmark detection
const CAMERA_CONSTRAINTS_HD = {
  video: {
    width: { ideal: 1920, min: 1280 },
    height: { ideal: 1080, min: 720 },
    frameRate: { ideal: 30, min: 15 },
  },
  audio: false,
};

// Fallback for mobile or lower-end cameras
const CAMERA_CONSTRAINTS_FALLBACK = {
  video: {
    width: { ideal: 1280, min: 640 },
    height: { ideal: 720, min: 480 },
    frameRate: { ideal: 30, min: 15 },
  },
  audio: false,
};

// ── Skeleton drawing colors ────────────────────────────────
const LANDMARK_COLOR = '#14b8a6';
const CONNECTION_COLOR = 'rgba(20, 184, 166, 0.4)';
const LANDMARK_RADIUS = 4;

// ── State ──────────────────────────────────────────────────
let poseLandmarker = null;
let drawingUtils = null;
let isModelLoaded = false;
let isCameraActive = false;
let isRecording = false;
let animFrameId = null;
let lastVideoTime = -1;
let frameCount = 0;
let fpsStartTime = 0;
let currentFps = 0;
let currentStream = null;
let facingMode = 'user'; // 'user' = front, 'environment' = rear
let currentDeviceId = null;

// Recorded pose data
let recordedFrames = [];
let recordingStartFrame = 0;

// Callbacks
let onLandmarksCallback = null;
let onFpsCallback = null;
let onModelLoadedCallback = null;
let onModelProgressCallback = null;

// ── Model Loading ──────────────────────────────────────────

async function loadModel() {
  Logger.info('Loading MediaPipe Pose Landmarker model...');

  try {
    if (onModelProgressCallback) onModelProgressCallback(10, 'Loading WASM runtime...');

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
    );

    if (onModelProgressCallback) onModelProgressCallback(40, 'Downloading pose model...');

    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_URL,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
    });

    isModelLoaded = true;

    if (onModelProgressCallback) onModelProgressCallback(100, 'Model ready!');
    if (onModelLoadedCallback) onModelLoadedCallback();

    Logger.info('Pose Landmarker model loaded successfully.');
    return true;
  } catch (err) {
    Logger.error(`Model loading failed: ${err.message}`);
    throw new Error(`Failed to load pose model: ${err.message}`);
  }
}

// ── Camera Management ──────────────────────────────────────

async function enumerateCameras() {
  try {
    // Need to request permission first to get labels
    const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
    tempStream.getTracks().forEach(t => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    Logger.info(`Found ${cameras.length} camera(s).`);
    return cameras;
  } catch (err) {
    Logger.warn(`Camera enumeration failed: ${err.message}`);
    return [];
  }
}

async function startCamera(videoElement, canvasElement, deviceId = null) {
  if (!isModelLoaded) {
    throw new Error('Pose model not loaded yet. Please wait for model initialization.');
  }

  Logger.info(`Starting camera (device: ${deviceId || 'default'}, facing: ${facingMode})...`);

  try {
    // Stop existing stream
    stopCamera();

    // Build constraints — request HIGH RESOLUTION for accurate landmarks
    let constraints;
    if (deviceId) {
      constraints = {
        video: {
          deviceId: { exact: deviceId },
          width: { ideal: 1920, min: 1280 },
          height: { ideal: 1080, min: 720 },
          frameRate: { ideal: 30, min: 15 },
        },
        audio: false,
      };
    } else {
      // Try HD first, fall back to lower res
      constraints = { ...CAMERA_CONSTRAINTS_HD };
      constraints.video = { ...constraints.video, facingMode: { ideal: facingMode } };
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (hdErr) {
      Logger.warn(`HD camera failed, trying fallback: ${hdErr.message}`);
      const fallback = { ...CAMERA_CONSTRAINTS_FALLBACK };
      fallback.video = { ...fallback.video, facingMode: { ideal: facingMode } };
      stream = await navigator.mediaDevices.getUserMedia(fallback);
    }

    currentStream = stream;
    videoElement.srcObject = stream;

    // Wait for video to be ready
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve();
      };
    });

    // Log actual resolution obtained
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    Logger.info(`Camera active: ${settings.width}×${settings.height} @ ${settings.frameRate || '?'}fps`);
    Logger.info(`Device: ${track.label}`);

    // Setup canvas to match video resolution
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;

    // Init drawing utils
    const ctx = canvasElement.getContext('2d');
    drawingUtils = new DrawingUtils(ctx);

    // Start detection loop
    isCameraActive = true;
    lastVideoTime = -1;
    frameCount = 0;
    fpsStartTime = performance.now();

    detectLoop(videoElement, canvasElement);

    return {
      width: settings.width || videoElement.videoWidth,
      height: settings.height || videoElement.videoHeight,
      fps: settings.frameRate,
      label: track.label,
    };
  } catch (err) {
    Logger.error(`Camera start failed: ${err.message}`);
    throw new Error(`Could not access camera: ${err.message}`);
  }
}

function stopCamera() {
  isCameraActive = false;

  if (animFrameId) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }

  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }

  Logger.info('Camera stopped.');
}

function flipCamera() {
  facingMode = facingMode === 'user' ? 'environment' : 'user';
  Logger.info(`Camera facing mode: ${facingMode}`);
  return facingMode;
}

// ── Detection Loop ─────────────────────────────────────────

function detectLoop(videoElement, canvasElement) {
  if (!isCameraActive || !poseLandmarker) return;

  const ctx = canvasElement.getContext('2d');
  const nowMs = performance.now();

  if (videoElement.currentTime !== lastVideoTime && videoElement.readyState >= 2) {
    lastVideoTime = videoElement.currentTime;

    try {
      const result = poseLandmarker.detectForVideo(videoElement, nowMs);
      drawResults(ctx, canvasElement, result, videoElement);
      processResults(result, frameCount);
      frameCount++;

      // FPS calculation
      const elapsed = (nowMs - fpsStartTime) / 1000;
      if (elapsed >= 1) {
        currentFps = Math.round(frameCount / elapsed);
        if (onFpsCallback) onFpsCallback(currentFps);
        frameCount = 0;
        fpsStartTime = nowMs;
      }
    } catch (err) {
      // Silently skip frame errors (can happen during transitions)
      if (!err.message?.includes('timestamp')) {
        Logger.warn(`Detection error: ${err.message}`);
      }
    }
  }

  animFrameId = requestAnimationFrame(() => detectLoop(videoElement, canvasElement));
}

// ── Drawing ────────────────────────────────────────────────

function drawResults(ctx, canvas, result, videoElement) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!result.landmarks || result.landmarks.length === 0) return;

  const landmarks = result.landmarks[0];

  // Draw connections (skeleton lines)
  for (const conn of PoseLandmarker.POSE_CONNECTIONS) {
    const from = landmarks[conn.start];
    const to = landmarks[conn.end];
    if (!from || !to) continue;
    if (from.visibility < VISIBILITY_THRESHOLD || to.visibility < VISIBILITY_THRESHOLD) continue;

    ctx.beginPath();
    ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
    ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
    ctx.strokeStyle = CONNECTION_COLOR;
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Draw landmarks
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    if (lm.visibility < VISIBILITY_THRESHOLD) continue;

    const x = lm.x * canvas.width;
    const y = lm.y * canvas.height;
    const isShoulder = i === LEFT_SHOULDER_IDX || i === RIGHT_SHOULDER_IDX;
    const isBody = BODY_INDICES.includes(i);

    ctx.beginPath();
    ctx.arc(x, y, isShoulder ? 6 : (isBody ? LANDMARK_RADIUS : 2.5), 0, Math.PI * 2);
    ctx.fillStyle = isShoulder ? '#f97316' : (isBody ? LANDMARK_COLOR : 'rgba(20,184,166,0.5)');
    ctx.fill();

    if (isShoulder) {
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // Draw shoulder width line
  const ls = landmarks[LEFT_SHOULDER_IDX];
  const rs = landmarks[RIGHT_SHOULDER_IDX];
  if (ls && rs && ls.visibility >= VISIBILITY_THRESHOLD && rs.visibility >= VISIBILITY_THRESHOLD) {
    ctx.beginPath();
    ctx.setLineDash([6, 4]);
    ctx.moveTo(ls.x * canvas.width, ls.y * canvas.height);
    ctx.lineTo(rs.x * canvas.width, rs.y * canvas.height);
    ctx.strokeStyle = '#f97316';
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

// ── Result Processing ──────────────────────────────────────

function processResults(result, frame) {
  if (!result.landmarks || result.landmarks.length === 0) {
    if (onLandmarksCallback) onLandmarksCallback(null, frame);
    return;
  }

  const landmarks = result.landmarks[0];
  const visibleCount = landmarks.filter(l => l.visibility >= VISIBILITY_THRESHOLD).length;

  // Build pose data for this frame
  const poseData = [];
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    poseData.push({
      index: i,
      name: LANDMARK_NAMES[i],
      x: lm.x,
      y: lm.y,
      z: lm.z,
      visibility: lm.visibility,
      isBody: BODY_INDICES.includes(i),
    });
  }

  // Calculate shoulder width
  const ls = landmarks[LEFT_SHOULDER_IDX];
  const rs = landmarks[RIGHT_SHOULDER_IDX];
  let shoulderWidth = NaN;
  if (ls && rs && ls.visibility >= VISIBILITY_THRESHOLD && rs.visibility >= VISIBILITY_THRESHOLD) {
    shoulderWidth = Math.sqrt((rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2);
  }

  const frameData = {
    frame,
    landmarks: poseData,
    shoulderWidth,
    visibleCount,
    timestamp: performance.now(),
  };

  // Store if recording
  if (isRecording) {
    recordedFrames.push(frameData);
  }

  if (onLandmarksCallback) onLandmarksCallback(frameData, frame);
}

// ── Recording ──────────────────────────────────────────────

function startRecording() {
  recordedFrames = [];
  recordingStartFrame = frameCount;
  isRecording = true;
  Logger.info('Recording started.');
}

function stopRecording() {
  isRecording = false;
  Logger.info(`Recording stopped. ${recordedFrames.length} frames captured.`);
  return recordedFrames;
}

function getRecordedFrames() {
  return recordedFrames;
}

/**
 * Convert recorded frames to CSV-compatible row format for analysis.js
 */
function framesToCSV(frames, source = 'mediapipe_browser', trial = null) {
  if (!trial) {
    const now = new Date();
    trial = `capture_${now.toISOString().slice(0, 19).replace(/[T:]/g, '_')}`;
  }

  const rows = ['source,domain,trial,frame,joint,x,y,z'];
  for (const f of frames) {
    for (const lm of f.landmarks) {
      if (lm.visibility < VISIBILITY_THRESHOLD) continue;
      rows.push(`${source},normalized_2d,${trial},${f.frame},${lm.name},${lm.x.toFixed(8)},${lm.y.toFixed(8)},${lm.z.toFixed(8)}`);
    }
  }
  return rows.join('\n');
}

// ── Video File Processing ──────────────────────────────────

async function processVideoFile(videoElement, canvasElement, onProgress) {
  if (!isModelLoaded) throw new Error('Model not loaded.');

  Logger.info('Processing video file...');

  const ctx = canvasElement.getContext('2d');
  const frames = [];
  let frameIdx = 0;
  const duration = videoElement.duration;
  const targetFps = 30;
  const frameInterval = 1 / targetFps;

  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  // Set to VIDEO mode
  poseLandmarker.setOptions({ runningMode: 'VIDEO' });

  return new Promise((resolve, reject) => {
    let currentTime = 0;

    function processNextFrame() {
      if (currentTime >= duration) {
        // Switch back to VIDEO mode for camera
        Logger.info(`Video processing complete. ${frameIdx} frames extracted.`);
        resolve(frames);
        return;
      }

      videoElement.currentTime = currentTime;
    }

    videoElement.onseeked = () => {
      try {
        const result = poseLandmarker.detectForVideo(videoElement, currentTime * 1000);

        // Draw on canvas for preview
        ctx.drawImage(videoElement, 0, 0);
        if (result.landmarks && result.landmarks.length > 0) {
          drawResults(ctx, canvasElement, result, videoElement);
        }

        // Process landmarks
        if (result.landmarks && result.landmarks.length > 0) {
          const landmarks = result.landmarks[0];
          const poseData = [];
          for (let i = 0; i < landmarks.length; i++) {
            poseData.push({
              index: i,
              name: LANDMARK_NAMES[i],
              x: landmarks[i].x,
              y: landmarks[i].y,
              z: landmarks[i].z,
              visibility: landmarks[i].visibility,
              isBody: BODY_INDICES.includes(i),
            });
          }

          const ls = landmarks[LEFT_SHOULDER_IDX];
          const rs = landmarks[RIGHT_SHOULDER_IDX];
          let shoulderWidth = NaN;
          if (ls && rs && ls.visibility >= VISIBILITY_THRESHOLD && rs.visibility >= VISIBILITY_THRESHOLD) {
            shoulderWidth = Math.sqrt((rs.x - ls.x) ** 2 + (rs.y - ls.y) ** 2);
          }

          frames.push({
            frame: frameIdx,
            landmarks: poseData,
            shoulderWidth,
            visibleCount: landmarks.filter(l => l.visibility >= VISIBILITY_THRESHOLD).length,
            timestamp: currentTime * 1000,
          });
        }

        frameIdx++;
        currentTime += frameInterval;

        const progress = Math.min(100, (currentTime / duration) * 100);
        if (onProgress) onProgress(progress, frameIdx, duration);

        // Continue — use setTimeout to yield to UI thread
        setTimeout(processNextFrame, 0);
      } catch (err) {
        Logger.warn(`Frame ${frameIdx} error: ${err.message}`);
        frameIdx++;
        currentTime += frameInterval;
        setTimeout(processNextFrame, 0);
      }
    };

    processNextFrame();
  });
}

// ── Public API ─────────────────────────────────────────────

const PoseEngine = {
  loadModel,
  enumerateCameras,
  startCamera,
  stopCamera,
  flipCamera,
  startRecording,
  stopRecording,
  getRecordedFrames,
  framesToCSV,
  processVideoFile,

  get isModelLoaded() { return isModelLoaded; },
  get isCameraActive() { return isCameraActive; },
  get isRecording() { return isRecording; },
  get currentFps() { return currentFps; },
  get recordedFrameCount() { return recordedFrames.length; },

  set onLandmarks(fn) { onLandmarksCallback = fn; },
  set onFps(fn) { onFpsCallback = fn; },
  set onModelLoaded(fn) { onModelLoadedCallback = fn; },
  set onModelProgress(fn) { onModelProgressCallback = fn; },

  LANDMARK_NAMES,
  BODY_INDICES,
  LEFT_SHOULDER_IDX,
  RIGHT_SHOULDER_IDX,
};

window.PoseEngine = PoseEngine;
export default PoseEngine;
