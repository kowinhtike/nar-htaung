// SPDX-License-Identifier: Apache-2.0
// Audio front-end port of Nara SAR `transcribe.py`: mono downmix, 16 kHz
// resampling, quiet-point chunking (8-10 s), and per-chunk zero-mean /
// unit-variance normalisation with epsilon 1e-5.
// Modified 2026: translated from Python to JavaScript; libsndfile replaced by
// the Web Audio API for container decoding.

export const SAMPLE_RATE = 16000;
export const MAX_CHUNK = 160000; // 10 s
export const MIN_CHUNK = 128000; // 8 s

export function downmixToMono(channels, length) {
  const mono = new Float32Array(length);
  const count = channels.length;
  if (count === 1) {
    mono.set(channels[0].subarray(0, length));
    return mono;
  }
  for (let i = 0; i < length; i++) {
    let sum = 0;
    for (let c = 0; c < count; c++) sum += channels[c][i];
    mono[i] = sum / count;
  }
  return mono;
}

// Linear interpolation, same formula as numpy in transcribe.py.
export function resampleLinear(mono, fromRate, toRate = SAMPLE_RATE) {
  if (fromRate === toRate) return mono;
  const outLength = Math.floor((mono.length * toRate) / fromRate);
  const out = new Float32Array(outLength);
  const step = fromRate / toRate;
  for (let i = 0; i < outLength; i++) {
    const position = i * step;
    const low = Math.floor(position);
    const fraction = position - low;
    const next = Math.min(low + 1, mono.length - 1);
    out[i] = mono[low] * (1 - fraction) + mono[next] * fraction;
  }
  return out;
}

export function chunkRanges(samples) {
  const ranges = [];
  const total = samples.length;
  let start = 0;
  while (start < total) {
    let end = Math.min(start + MAX_CHUNK, total);
    if (end < total) {
      const limit = end;
      let quietest = Infinity;
      for (let at = limit - 32000; at <= limit - 3200; at += 320) {
        let energy = 0;
        for (let i = at; i < at + 3200; i++) {
          const x = samples[i];
          energy += x * x;
        }
        if (energy < quietest) {
          quietest = energy;
          end = Math.max(start + MIN_CHUNK, at + 1600);
        }
      }
    }
    if (total - end < 4000) end = total;
    ranges.push([start, end]);
    start = end;
  }
  return ranges;
}

export function normalizeChunk(samples, start, end) {
  const length = end - start;
  const values = new Float64Array(length);
  let sum = 0;
  for (let i = 0; i < length; i++) sum += samples[start + i];
  const mean = sum / length;
  let squares = 0;
  for (let i = 0; i < length; i++) {
    const v = samples[start + i] - mean;
    values[i] = v;
    squares += v * v;
  }
  const scale = Math.sqrt(squares / length + 1e-5);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) out[i] = values[i] / scale;
  return out;
}

export function peak(samples, start, end) {
  let max = 0;
  for (let i = start; i < end; i++) {
    const v = Math.abs(samples[i]);
    if (v > max) max = v;
  }
  return max;
}

// Browser-only: decode any browser-supported audio container to 16 kHz mono.
export async function decodeAudioFile(arrayBuffer) {
  const Ctx = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
  if (!Ctx) throw new Error('Web Audio is not available in this browser');
  const context = new Ctx(1, 1, SAMPLE_RATE);
  const buffer = await context.decodeAudioData(arrayBuffer.slice(0));
  const channels = [];
  for (let c = 0; c < buffer.numberOfChannels; c++) channels.push(buffer.getChannelData(c));
  const mono = downmixToMono(channels, buffer.length);
  return { samples: resampleLinear(mono, buffer.sampleRate), sourceRate: buffer.sampleRate, duration: buffer.duration };
}
