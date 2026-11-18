// SPDX-License-Identifier: Apache-2.0
// Port of the Nara SAR compact bilingual character n-gram language model
// ("NARALM01" container) from `sar_decoder.py`.
// Modified 2026: translated from Python to JavaScript for browser execution.
//
// Container layout (all little-endian):
//   0..7    magic "NARALM01"
//   8..11   uint32 header byte length
//   12..    header JSON  { format, keyWidth, vocab: string[], blocks: [{count, offset}] }
//   start   records: key (order * keyWidth bytes) + float32 logprob + float32 backoff
// Records inside each order block are sorted lexicographically by key bytes,
// which is what the binary search below relies on.

const MAGIC = 'NARALM01';

export class LanguageModel {
  constructor(buffer) {
    this.u8 = new Uint8Array(buffer);
    this.dv = new DataView(buffer);
    let magic = '';
    for (let i = 0; i < 8; i++) magic += String.fromCharCode(this.u8[i]);
    if (magic !== MAGIC) throw new Error('Invalid SAR language model');

    const headerLength = this.dv.getUint32(8, true);
    const headerBytes = this.u8.subarray(12, 12 + headerLength);
    this.header = JSON.parse(new TextDecoder().decode(headerBytes));
    this.start = 12 + headerLength;

    this.keyWidth = this.header.keyWidth;
    this.blocks = this.header.blocks;

    this.vocab = new Map();
    this.header.vocab.forEach((word, index) => this.vocab.set(word, index));
    this.unknown = this.vocab.get('<unk>');
    this.bos = [this.vocab.get('<s>')];

    this.keyScratch = null;
    this.advanceCache = new Map();
  }

  lookup(ids) {
    const order = ids.length;
    if (!order) return null;
    const block = this.blocks[order - 1];
    const width = this.keyWidth;
    const keyLength = order * width;
    const recordSize = keyLength + 8;
    const first = this.start + block.offset;
    const u8 = this.u8;

    let key = this.keyScratch;
    if (!key || key.length < keyLength) {
      key = this.keyScratch = new Uint8Array(Math.max(keyLength, 8));
    }
    for (let i = 0; i < order; i++) {
      let value = ids[i];
      for (let b = width - 1; b >= 0; b--) {
        key[i * width + b] = value & 0xff;
        value = Math.floor(value / 256);
      }
    }

    let low = 0;
    let high = block.count;
    while (low < high) {
      const middle = (low + high) >>> 1;
      const at = first + middle * recordSize;
      let cmp = 0;
      for (let i = 0; i < keyLength; i++) {
        const a = u8[at + i];
        const b = key[i];
        if (a !== b) {
          cmp = a < b ? -1 : 1;
          break;
        }
      }
      if (cmp < 0) low = middle + 1;
      else if (cmp > 0) high = middle;
      else return [this.dv.getFloat32(at + keyLength, true), this.dv.getFloat32(at + keyLength + 4, true)];
    }
    return null;
  }

  advance(state, char) {
    const cacheKey = state.join(',') + '|' + char;
    const hit = this.advanceCache.get(cacheKey);
    if (hit !== undefined) return hit;

    const word = char === ' ' ? '▁' : char;
    const index = this.vocab.has(word) ? this.vocab.get(word) : this.unknown;
    let history = state;
    let backoff = 0;
    let score;
    for (;;) {
      const found = this.lookup(history.concat([index]));
      if (found !== null) {
        score = backoff + found[0];
        break;
      }
      if (!history.length) throw new Error('Missing SAR language-model unigram');
      const context = this.lookup(history);
      if (context !== null) backoff += context[1];
      history = history.slice(1);
    }
    const next = state.concat([index]).slice(-5);
    const out = [score, next];
    if (this.advanceCache.size > 300000) this.advanceCache.clear();
    this.advanceCache.set(cacheKey, out);
    return out;
  }
}
