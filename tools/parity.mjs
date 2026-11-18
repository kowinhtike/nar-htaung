// SPDX-License-Identifier: Apache-2.0
// Parity check: the JavaScript port against a Python reference run.
//
//   python tools/dump_reference.py samples/sample.wav /tmp/ref_out
//   node tools/parity.mjs /tmp/ref_out
//
// Checks, in order:
//   1. chunk boundaries and normalised chunk input (bit-level tolerance)
//   2. beam decoder output on identical logits (must match exactly)
//   3. end-to-end ONNX Runtime Web run on the same chunk (text + logit drift)

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';

import { chunkRanges, normalizeChunk } from '../src/audio.js';
import { LanguageModel } from '../src/lm.js';
import { Decoder } from '../src/decoder.js';

const dir = process.argv[2];
if (!dir) {
  console.error('usage: node tools/parity.mjs <reference-dir>');
  process.exit(1);
}

const readF32 = (name) => {
  const buffer = readFileSync(join(dir, name));
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
const sarJson = JSON.parse(readFileSync(new URL('../models/sar.json', import.meta.url), 'utf8'));
const lmBuffer = readFileSync(new URL('../models/language.lm', import.meta.url));
const lm = new LanguageModel(lmBuffer.buffer.slice(lmBuffer.byteOffset, lmBuffer.byteOffset + lmBuffer.byteLength));
const decoder = new Decoder({ sarJson, lm });

const mono = readF32('mono.f32');
const ranges = chunkRanges(mono);

let failures = 0;
const check = (label, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}${detail ? `  ${detail}` : ''}`);
  if (!ok) failures++;
};

console.log(`audio ${meta.audio_seconds.toFixed(2)}s  chunks(py)=${meta.chunks.length} chunks(js)=${ranges.length}`);

// 1. chunking + normalisation
let boundaryOk = ranges.length === meta.chunks.length;
for (let i = 0; i < Math.min(ranges.length, meta.chunks.length); i++) {
  if (ranges[i][0] !== meta.chunks[i].start || ranges[i][1] !== meta.chunks[i].end) boundaryOk = false;
}
check('chunk boundaries match', boundaryOk);

let maxInputDiff = 0;
for (let i = 0; i < Math.min(ranges.length, meta.chunks.length); i++) {
  const js = normalizeChunk(mono, ranges[i][0], ranges[i][1]);
  const ref = readF32(`chunk${i}.f32`);
  if (js.length !== ref.length) {
    maxInputDiff = Infinity;
    break;
  }
  for (let j = 0; j < js.length; j++) maxInputDiff = Math.max(maxInputDiff, Math.abs(js[j] - ref[j]));
}
check('normalised input matches reference', maxInputDiff < 1e-4, `max|diff|=${maxInputDiff.toExponential(2)}`);

// 2. decoder parity on the reference logits
const decodeTexts = [];
for (let i = 0; i < meta.chunks.length; i++) {
  const logits = readF32(`chunk${i}.logits.f32`);
  const frames = meta.chunks[i].frames;
  const text = decoder.decodeLogits(logits, frames).trim();
  decodeTexts.push(text);
  check(`decoder text chunk ${i}`, text === meta.chunks[i].text, `\n   js : ${JSON.stringify(text)}\n   py : ${JSON.stringify(meta.chunks[i].text)}`);
}

// 3. end-to-end with ONNX Runtime Web (WASM) on the same input
ort.env.wasm.numThreads = 2;
ort.env.logLevel = 'error';
ort.env.wasm.wasmPaths = new URL('../node_modules/onnxruntime-web/dist/', import.meta.url).pathname;

const modelBytes = readFileSync(new URL('../models/acoustic.onnx', import.meta.url));
const started = performance.now();
const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
});
console.log(`\nORT Web session ready in ${((performance.now() - started) / 1000).toFixed(1)}s`);

for (let i = 0; i < Math.min(ranges.length, meta.chunks.length); i++) {
  const input = normalizeChunk(mono, ranges[i][0], ranges[i][1]);
  const feed = { audio: new ort.Tensor('float32', input, [1, input.length]) };
  const t0 = performance.now();
  const output = await session.run(feed);
  const elapsed = (performance.now() - t0) / 1000;
  const logits = output.logits.data;
  const frames = output.logits.dims[1];
  const ref = readF32(`chunk${i}.logits.f32`);

  let maxLogitDiff = 0;
  let argmaxDisagree = 0;
  const vocab = meta.chunks[i].vocab;
  for (let f = 0; f < frames; f++) {
    let bestJs = 0;
    let bestPy = 0;
    for (let v = 0; v < vocab; v++) {
      const a = logits[f * vocab + v];
      const b = ref[f * vocab + v];
      const diff = Math.abs(a - b);
      if (diff > maxLogitDiff) maxLogitDiff = diff;
      if (a > logits[f * vocab + bestJs]) bestJs = v;
      if (b > ref[f * vocab + bestPy]) bestPy = v;
    }
    if (bestJs !== bestPy) argmaxDisagree++;
  }

  const text = decoder.decodeLogits(logits, frames).trim();
  const seconds = (ranges[i][1] - ranges[i][0]) / 16000;
  console.log(`\nchunk ${i}: ${seconds.toFixed(2)}s audio -> ${elapsed.toFixed(2)}s (RTF ${(elapsed / seconds).toFixed(2)}x)`);
  console.log(`  max|logit diff| = ${maxLogitDiff.toFixed(4)}  frame argmax disagreements = ${argmaxDisagree}/${frames}`);
  console.log(`  js : ${JSON.stringify(text)}`);
  console.log(`  py : ${JSON.stringify(meta.chunks[i].text)}`);
  check(`end-to-end text chunk ${i}`, text === meta.chunks[i].text);
}

console.log(`\nreference full text: ${JSON.stringify(meta.text)}`);
console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
