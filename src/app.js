// SPDX-License-Identifier: Apache-2.0
// UI shell for the standalone Nara SAR web client. All inference happens in
// the worker; nothing is uploaded.

import { decodeAudioFile } from './audio.js';

const $ = (id) => document.getElementById(id);

const els = {
  status: $('status'),
  statusText: $('statusText'),
  loader: $('loader'),
  loaderLabel: $('loaderLabel'),
  loaderPct: $('loaderPct'),
  loaderBar: $('loaderBar'),
  loaderHint: $('loaderHint'),
  errorBox: $('errorBox'),
  drop: $('drop'),
  pick: $('pick'),
  record: $('record'),
  fileInput: $('fileInput'),
  fileRow: $('fileRow'),
  fileName: $('fileName'),
  fileInfo: $('fileInfo'),
  remove: $('remove'),
  run: $('run'),
  stop: $('stop'),
  stats: $('stats'),
  transcript: $('transcript'),
  progress: $('progress'),
  chunkBar: $('chunkBar'),
  chunkLabel: $('chunkLabel'),
  copy: $('copy'),
  save: $('save'),
};

const state = {
  ready: false,
  busy: false,
  threads: 1,
  samples: null,
  fileName: '',
  duration: 0,
  result: '',
  recorder: null,
  recording: false,
  recordStart: 0,
  timer: null,
};

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

function setStatus(text, kind) {
  els.statusText.textContent = text;
  els.status.dataset.state = kind;
}

function idleStatus() {
  const label = `Ready · WASM ${state.threads} thread${state.threads > 1 ? 's' : ''}`;
  setStatus(state.ready ? label : 'Loading model…', state.ready ? 'ready' : 'busy');
}

function showError(message) {
  els.errorBox.textContent = message;
  els.errorBox.hidden = false;
}

function clearError() {
  els.errorBox.hidden = true;
  els.errorBox.textContent = '';
}

function setBusy(busy) {
  state.busy = busy;
  els.run.disabled = busy || !state.samples || !state.ready;
  els.pick.disabled = busy;
  els.record.disabled = busy;
  els.remove.disabled = busy;
  els.stop.hidden = !busy;
}

function formatBytes(bytes) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.round(bytes / 1024)} KB`;
}

function formatTime(seconds) {
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return minutes ? `${minutes}m ${String(rest).padStart(2, '0')}s` : `${rest}s`;
}

/* model ---------------------------------------------------------------- */

worker.onmessage = (event) => {
  const message = event.data;
  switch (message.type) {
    case 'load-progress': {
      const percent = Math.round(message.fraction * 100);
      els.loaderBar.style.width = `${percent}%`;
      els.loaderPct.textContent = `${percent}%`;
      els.loaderLabel.textContent = message.cached
        ? `Loading ${message.label} from cache…`
        : `Downloading ${message.label}…`;
      break;
    }
    case 'source': {
      if (message.source !== 'local') {
        els.loaderHint.textContent =
          `Model files are being fetched from ${message.label}. The first run downloads about 430 MB once and keeps it in this browser.`;
      }
      els.status.title = `Models: ${message.label}`;
      break;
    }
    case 'ready': {
      state.ready = true;
      state.threads = message.threads;
      els.loader.hidden = true;
      els.status.title = `Models: ${message.source}`;
      setStatus(`Ready · WASM ${message.threads} thread${message.threads > 1 ? 's' : ''}`, 'ready');
      updateRunState();
      break;
    }
    case 'chunk': {
      const fraction = (message.index + 1) / message.total;
      els.chunkBar.style.width = `${Math.round(fraction * 100)}%`;
      els.chunkLabel.textContent = `${message.index + 1} / ${message.total}`;
      renderTranscript(message.partial, true);
      break;
    }
    case 'done': {
      state.result = message.text;
      renderTranscript(message.text, false);
      els.progress.hidden = true;
      const stats = message.stats;
      els.stats.textContent =
        `${formatTime(stats.audioSeconds)} audio · ${stats.chunks} chunk${stats.chunks > 1 ? 's' : ''} · ` +
        `${stats.elapsed.toFixed(1)}s compute · RTF ${stats.rtf.toFixed(2)}×`;
      els.copy.disabled = !message.text;
      els.save.disabled = !message.text;
      setBusy(false);
      idleStatus();
      break;
    }
    case 'cancelled': {
      els.progress.hidden = true;
      setBusy(false);
      idleStatus();
      break;
    }
    case 'error': {
      showError(message.message);
      setBusy(false);
      setStatus('Error', 'error');
      els.progress.hidden = true;
      break;
    }
    default:
      break;
  }
};

worker.onerror = (event) => {
  showError(event.message || 'Worker failed to start.');
  setStatus('Error', 'error');
};

/* transcript ----------------------------------------------------------- */

function renderTranscript(text, partial) {
  els.transcript.dataset.empty = text ? 'false' : 'true';
  if (!text) {
    els.transcript.textContent = partial ? 'Listening…' : 'Your transcript will appear here.';
    return;
  }
  els.transcript.textContent = text;
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

/* audio input ---------------------------------------------------------- */

const MAX_BYTES = 256 * 1024 * 1024;
const MAX_SECONDS = 1800;

async function useFile(file) {
  clearError();
  if (file.size > MAX_BYTES) {
    showError(`File is ${formatBytes(file.size)}. The limit is 256 MB.`);
    return;
  }
  setStatus('Decoding audio…', 'busy');
  try {
    const buffer = await file.arrayBuffer();
    const decoded = await decodeAudioFile(buffer);
    if (decoded.duration < 0.25) {
      showError('Audio must be at least 0.25 seconds long.');
      idleStatus();
      return;
    }
    if (decoded.duration > MAX_SECONDS) {
      showError(`Audio is ${formatTime(decoded.duration)}. The limit is 30 minutes.`);
      idleStatus();
      return;
    }
    state.samples = decoded.samples;
    state.fileName = file.name || 'recording.wav';
    state.duration = decoded.duration;
    state.result = '';
    els.fileName.textContent = state.fileName;
    els.fileInfo.textContent = `${formatTime(decoded.duration)} · ${decoded.sourceRate} Hz source · 16 kHz mono`;
    els.fileRow.hidden = false;
    els.stats.textContent = '';
    els.copy.disabled = true;
    els.save.disabled = true;
    renderTranscript('', false);
    idleStatus();
    updateRunState();
  } catch (error) {
    showError(`Could not decode this audio file: ${error.message}`);
    setStatus('Error', 'error');
  }
}

function updateRunState() {
  els.run.disabled = state.busy || !state.samples || !state.ready;
}

function resetFile() {
  state.samples = null;
  state.fileName = '';
  state.duration = 0;
  state.result = '';
  els.fileRow.hidden = true;
  els.stats.textContent = '';
  els.copy.disabled = true;
  els.save.disabled = true;
  renderTranscript('', false);
  updateRunState();
}

els.pick.addEventListener('click', () => els.fileInput.click());
els.fileInput.addEventListener('change', () => {
  const file = els.fileInput.files && els.fileInput.files[0];
  if (file) useFile(file);
  els.fileInput.value = '';
});
els.remove.addEventListener('click', resetFile);

['dragenter', 'dragover'].forEach((type) => {
  els.drop.addEventListener(type, (event) => {
    event.preventDefault();
    els.drop.classList.add('drag');
  });
});
['dragleave', 'drop'].forEach((type) => {
  els.drop.addEventListener(type, (event) => {
    event.preventDefault();
    els.drop.classList.remove('drag');
  });
});
els.drop.addEventListener('drop', (event) => {
  const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) useFile(file);
});
document.addEventListener('dragover', (event) => event.preventDefault());
document.addEventListener('drop', (event) => event.preventDefault());

/* microphone ----------------------------------------------------------- */

function pickMimeType() {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  for (const type of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
  }
  return '';
}

async function toggleRecording() {
  if (state.recording) {
    state.recorder.stop();
    return;
  }
  clearError();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
    const mimeType = pickMimeType();
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunks.push(event.data);
    };
    recorder.onstop = async () => {
      clearInterval(state.timer);
      stream.getTracks().forEach((track) => track.stop());
      state.recording = false;
      els.record.classList.remove('recording');
      els.record.textContent = 'Record mic';
      const blob = new Blob(chunks, { type: mimeType || 'audio/webm' });
      if (!blob.size) {
        showError('Nothing was recorded.');
        idleStatus();
        return;
      }
      const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
      const file = new File([blob], `recording-${stamp}.webm`, { type: blob.type });
      await useFile(file);
    };
    state.recorder = recorder;
    state.recording = true;
    state.recordStart = performance.now();
    recorder.start();
    els.record.classList.add('recording');
    els.record.textContent = 'Stop (0s)';
    setStatus('Recording…', 'busy');
    state.timer = setInterval(() => {
      const seconds = Math.round((performance.now() - state.recordStart) / 1000);
      els.record.textContent = `Stop (${seconds}s)`;
      if (seconds >= MAX_SECONDS) recorder.stop();
    }, 500);
  } catch (error) {
    showError(`Microphone unavailable: ${error.message}`);
    idleStatus();
  }
}

els.record.addEventListener('click', toggleRecording);

/* run ------------------------------------------------------------------ */

els.run.addEventListener('click', () => {
  if (!state.samples || !state.ready || state.busy) return;
  clearError();
  setBusy(true);
  setStatus('Transcribing…', 'busy');
  els.progress.hidden = false;
  els.chunkBar.style.width = '0%';
  els.chunkLabel.textContent = '0 / –';
  els.stats.textContent = '';
  els.copy.disabled = true;
  els.save.disabled = true;
  renderTranscript('', true);
  // Send a copy so the loaded audio stays available for a re-run.
  const samples = state.samples.slice();
  worker.postMessage({ type: 'transcribe', samples }, [samples.buffer]);
});

els.stop.addEventListener('click', () => worker.postMessage({ type: 'cancel' }));

els.copy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(state.result);
    const label = els.copy.textContent;
    els.copy.textContent = 'Copied';
    setTimeout(() => (els.copy.textContent = label), 1200);
  } catch {
    showError('Clipboard access was blocked by the browser.');
  }
});

els.save.addEventListener('click', () => {
  const blob = new Blob([state.result + '\n'], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = (state.fileName.replace(/\.[^.]+$/, '') || 'transcript') + '.txt';
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

/* boot ----------------------------------------------------------------- */

if (!window.crossOriginIsolated) {
  els.loaderHint.innerHTML =
    'First run downloads about 430 MB once. This page is not cross-origin isolated, so inference runs ' +
    'single-threaded and will be slower — serve it with <code>npm start</code> to enable threads.';
}

setStatus('Loading model…', 'busy');
const sourcePreference = new URLSearchParams(location.search).get('source') || undefined;
worker.postMessage({ type: 'load', source: sourcePreference });
