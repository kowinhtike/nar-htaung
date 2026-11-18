// SPDX-License-Identifier: Apache-2.0
// Inference worker: loads the Nara SAR assets once, runs the INT8 acoustic
// model with ONNX Runtime Web (WASM) and the ported beam decoder.
// Modified 2026: JavaScript port of the Python reference runner for browsers.

import * as ort from '../vendor/ort/ort.wasm.min.mjs';
import { chunkRanges, normalizeChunk, peak } from './audio.js';
import { LanguageModel } from './lm.js';
import { Decoder } from './decoder.js';

const CACHE_NAME = 'nara-sar-v1';

// Model files can be served from this origin (./models, the fast path) or
// fetched straight from the Hugging Face CDN, so a fresh clone works even
// without the 430 MB of weights on disk.
const SOURCES = {
  local: {
    base: new URL('../models/', import.meta.url).href,
    probe: 'sar.json',
    paths: { sarJson: 'sar.json', lm: 'language.lm', onnx: 'acoustic.onnx' },
    label: 'this server',
  },
  hf: {
    base: 'https://huggingface.co/YeeNyunn/nara-sar/resolve/main/',
    probe: 'sar.json',
    paths: { sarJson: 'sar.json', lm: 'language.lm', onnx: 'onnx/acoustic.onnx' },
    label: 'Hugging Face',
  },
};

const ASSETS = [
  { key: 'sarJson', bytes: 185781, label: 'decoder config', file: 'sar.json' },
  { key: 'lm', bytes: 83704940, label: 'language model', file: 'language.lm' },
  { key: 'onnx', bytes: 365906429, label: 'acoustic model', file: 'acoustic.onnx' },
];

let session = null;
let decoder = null;
let cancelled = false;
let activeSource = 'local';

const absolute = (path) => new URL(path, import.meta.url).href;
const assetUrl = (asset) => SOURCES[activeSource].base + SOURCES[activeSource].paths[asset.key];

async function chooseSource(preference) {
  if (preference === 'local' || preference === 'hf') {
    activeSource = preference;
    return;
  }
  try {
    const response = await fetch(SOURCES.local.base + SOURCES.local.probe, { method: 'HEAD' });
    activeSource = response.ok ? 'local' : 'hf';
  } catch {
    activeSource = 'hf';
  }
}

// Cache Storage refuses single entries this large in Chrome ("Unexpected
// internal error"), so the model files live in the origin private file system
// instead. Cache Storage stays as a fallback for browsers without OPFS.
async function openStorage() {
  const store = { opfs: null, cache: null };
  try {
    if (self.navigator.storage && self.navigator.storage.getDirectory) {
      store.opfs = await self.navigator.storage.getDirectory();
    }
  } catch {
    store.opfs = null;
  }
  if (!store.opfs) {
    try {
      if ('caches' in self) store.cache = await caches.open(CACHE_NAME);
    } catch {
      store.cache = null;
    }
  }
  return store;
}

async function readCached(store, asset) {
  if (store.opfs) {
    try {
      const handle = await store.opfs.getFileHandle(asset.file);
      const file = await handle.getFile();
      if (file.size !== asset.bytes) return null;
      return new Uint8Array(await file.arrayBuffer());
    } catch {
      return null;
    }
  }
  if (store.cache) {
    const hit = await store.cache.match(assetUrl(asset));
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  return null;
}

async function downloadAsset(store, asset, onProgress) {
  const url = assetUrl(asset);
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);

  let writable = null;
  if (store.opfs) {
    try {
      const handle = await store.opfs.getFileHandle(asset.file, { create: true });
      writable = await handle.createWritable();
    } catch {
      writable = null;
    }
  } else if (store.cache) {
    store.cache.put(url, response.clone()).catch(() => {});
  }

  const total = Number(response.headers.get('content-length')) || asset.bytes;
  const reader = response.body.getReader();
  const bytes = new Uint8Array(asset.bytes);
  let received = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (received + value.length > bytes.length) throw new Error(`${asset.file} is larger than expected`);
      bytes.set(value, received);
      if (writable) await writable.write(value);
      received += value.length;
      onProgress(received, total);
    }
    if (writable) await writable.close();
  } catch (error) {
    if (writable) {
      try {
        await writable.abort();
      } catch {}
      try {
        await store.opfs.removeEntry(asset.file);
      } catch {}
    }
    throw error;
  }

  if (received !== asset.bytes) {
    throw new Error(`${asset.file} was truncated (${received} of ${asset.bytes} bytes)`);
  }
  return bytes;
}

async function loadModel(preference) {
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, Math.max(1, self.navigator.hardwareConcurrency || 2))
    : 1;
  ort.env.wasm.simd = true;
  ort.env.logLevel = 'error';
  ort.env.wasm.wasmPaths = absolute('../vendor/ort/');

  if (self.navigator.storage && self.navigator.storage.persist) {
    self.navigator.storage.persist().catch(() => {});
  }

  await chooseSource(preference);
  self.postMessage({ type: 'source', source: activeSource, label: SOURCES[activeSource].label });
  const store = await openStorage();
  const totals = ASSETS.reduce((sum, asset) => sum + asset.bytes, 0);
  let done = 0;
  const loaded = {};

  for (const asset of ASSETS) {
    const before = done;
    let bytes = await readCached(store, asset);
    const cached = bytes !== null;
    if (cached) {
      self.postMessage({
        type: 'load-progress',
        label: asset.label,
        fileFraction: 1,
        fraction: Math.min(1, (before + asset.bytes) / totals),
        cached: true,
      });
    } else {
      bytes = await downloadAsset(store, asset, (received, total) => {
        const fraction = total ? Math.min(1, received / total) : 0;
        self.postMessage({
          type: 'load-progress',
          label: asset.label,
          fileFraction: fraction,
          fraction: Math.min(1, (before + fraction * asset.bytes) / totals),
          cached: false,
        });
      });
    }
    done = before + asset.bytes;
    loaded[asset.key] = bytes;
    self.postMessage({
      type: 'load-progress',
      label: asset.label,
      fileFraction: 1,
      fraction: Math.min(1, done / totals),
      cached,
    });
  }

  const sarJson = JSON.parse(new TextDecoder().decode(loaded.sarJson));
  const lmBuffer = loaded.lm.buffer.slice(loaded.lm.byteOffset, loaded.lm.byteOffset + loaded.lm.byteLength);
  const lm = new LanguageModel(lmBuffer);
  decoder = new Decoder({ sarJson, lm });

  const started = performance.now();
  session = await ort.InferenceSession.create(loaded.onnx, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  const sessionMs = performance.now() - started;

  self.postMessage({
    type: 'ready',
    threads: ort.env.wasm.numThreads,
    isolated: Boolean(self.crossOriginIsolated),
    source: SOURCES[activeSource].label,
    sessionMs,
  });
}

async function transcribe(samples) {
  if (!session || !decoder) throw new Error('Model is not loaded yet');
  cancelled = false;
  const ranges = chunkRanges(samples);
  const parts = [];
  const started = performance.now();

  for (let index = 0; index < ranges.length; index++) {
    if (cancelled) {
      self.postMessage({ type: 'cancelled' });
      return;
    }
    const [start, end] = ranges[index];
    const input = normalizeChunk(samples, start, end);
    let text = '';
    if (peak(input, 0, input.length) > 1e-5) {
      const feed = { audio: new ort.Tensor('float32', input, [1, input.length]) };
      const output = await session.run(feed);
      text = decoder.decodeLogits(output.logits.data, output.logits.dims[1]).trim();
    }
    if (text) parts.push(text);
    self.postMessage({
      type: 'chunk',
      index,
      total: ranges.length,
      text,
      partial: parts.join(' '),
      seconds: (end - start) / 16000,
    });
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const elapsed = (performance.now() - started) / 1000;
  self.postMessage({
    type: 'done',
    text: parts.join(' '),
    stats: {
      audioSeconds: samples.length / 16000,
      chunks: ranges.length,
      elapsed,
      rtf: elapsed / (samples.length / 16000 || 1),
    },
  });
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === 'load') {
      await loadModel(message.source);
    } else if (message.type === 'transcribe') {
      await transcribe(message.samples);
    } else if (message.type === 'cancel') {
      cancelled = true;
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: String(error && error.message ? error.message : error) });
  }
};
