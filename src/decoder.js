// SPDX-License-Identifier: Apache-2.0
// Port of Nara SAR `sar_decoder.py`: CTC prefix beam search with the bilingual
// character n-gram LM and the Burmese structural syllable gate.
// Modified 2026: translated from Python to JavaScript for browser execution.
//
// Fixed final decoding recipe (see sar.json): beam 16, top-k 10, alpha 0.6,
// beta 0, blank id 0. Greedy argmax is a different recipe and is intentionally
// not offered here.

import { SyllableConstraint } from './syllables.js';

const BEAM = 16;
const TOPK = 10;
const ALPHA = 0.6;

function logsum(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const high = Math.max(a, b);
  return high + Math.log(Math.exp(a - high) + Math.exp(b - high));
}

// Top-k with a stable tie-break: equal logits keep the smaller token id first,
// matching numpy's stable argsort on the negated row.
function topKRow(logits, offset, vocabSize, outIds, outVals, k = TOPK) {
  let n = 0;
  for (let i = 0; i < vocabSize; i++) {
    const v = logits[offset + i];
    if (v !== v) continue; // NaN never ranks
    if (n === k && v <= outVals[k - 1]) continue;
    let pos = n < k ? n : k - 1;
    while (pos > 0 && v > outVals[pos - 1]) {
      outVals[pos] = outVals[pos - 1];
      outIds[pos] = outIds[pos - 1];
      pos--;
    }
    outVals[pos] = v;
    outIds[pos] = i;
    if (n < k) n++;
  }
  return n;
}

export class Decoder {
  constructor({ sarJson, lm }) {
    this.lm = lm;
    this.gate = new SyllableConstraint();
    this.vocabSize = sarJson.vocabSize;
    this.id2char = new Array(sarJson.vocabSize);
    for (const [key, value] of Object.entries(sarJson.id2char)) {
      this.id2char[Number(key)] = value;
    }
  }

  // rows: iterable of arrays of [tokenId, logit] pairs (already top-k).
  decodeRows(rows) {
    const gate = this.gate;
    const lm = this.lm;
    let beams = new Map();
    beams.set('', [0.0, -Infinity, 0.0, lm.bos, gate.startState()]);

    for (const row of rows) {
      const candidates = row.slice().sort((a, b) => b[1] - a[1]).slice(0, TOPK);
      if (!candidates.length) throw new Error('Empty SAR acoustic row');
      const shift = candidates[0][1];
      const following = new Map();
      const entry = (prefix) => {
        let item = following.get(prefix);
        if (item === undefined) {
          item = [-Infinity, -Infinity, 0.0, null, null];
          following.set(prefix, item);
        }
        return item;
      };

      for (const [prefix, beam] of beams) {
        const pb = beam[0];
        const pnb = beam[1];
        const lmScore = beam[2];
        const state = beam[3];
        const cs = beam[4];
        const total = logsum(pb, pnb);
        const last = prefix.length ? prefix[prefix.length - 1] : null;

        for (let c = 0; c < candidates.length; c++) {
          const index = candidates[c][0];
          const lp = candidates[c][1] - shift; // common per-frame shift keeps CTC ranking
          if (index === 0) {
            const item = entry(prefix);
            item[0] = logsum(item[0], total + lp);
            item[2] = lmScore;
            item[3] = state;
            item[4] = cs;
            continue;
          }
          const char = this.id2char[index];
          if (char === undefined) continue;
          const nextCs = gate.step(cs, char);
          if (nextCs === null) continue;
          const advanced = lm.advance(state, char);
          const nextScore = advanced[0];
          const nextState = advanced[1];
          const nextPrefix = prefix + char;
          let probability;
          if (char === last) {
            const item = entry(prefix);
            item[1] = logsum(item[1], pnb + lp);
            item[2] = lmScore;
            item[3] = state;
            item[4] = cs;
            probability = pb + lp;
          } else {
            probability = total + lp;
          }
          const item = entry(nextPrefix);
          item[1] = logsum(item[1], probability);
          item[2] = lmScore + nextScore;
          item[3] = nextState;
          item[4] = nextCs;
        }
      }

      if (following.size === 0) throw new Error('SAR final decoder has no legal beam');

      const scored = [];
      for (const [prefix, item] of following) {
        scored.push([prefix, item, logsum(item[0], item[1]) + ALPHA * item[2]]);
      }
      scored.sort((a, b) => b[2] - a[2]);
      beams = new Map();
      for (let i = 0; i < scored.length && i < BEAM; i++) {
        beams.set(scored[i][0], scored[i][1]);
      }
    }

    let pool = [];
    for (const pair of beams) {
      if (gate.canFinish(pair[1][4])) pool.push(pair);
    }
    if (!pool.length) pool = [...beams];
    let best = '';
    let bestScore = -Infinity;
    for (const [text, item] of pool) {
      const score = logsum(item[0], item[1]) + ALPHA * item[2];
      if (score > bestScore) {
        bestScore = score;
        best = text;
      }
    }
    return best;
  }

  // logits: Float32Array of shape [frames, vocabSize] (flattened, one batch).
  decodeLogits(logits, frames) {
    const vocabSize = this.vocabSize;
    const outIds = new Int32Array(TOPK);
    const outVals = new Float32Array(TOPK);
    const rows = [];
    for (let f = 0; f < frames; f++) {
      const offset = f * vocabSize;
      const n = topKRow(logits, offset, vocabSize, outIds, outVals);
      const row = new Array(n);
      for (let i = 0; i < n; i++) row[i] = [outIds[i], outVals[i]];
      rows.push(row);
    }
    return this.decodeRows(rows);
  }
}
