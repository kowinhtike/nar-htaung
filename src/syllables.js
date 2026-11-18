// SPDX-License-Identifier: Apache-2.0
// Port of Nara SAR `sar_syllables.py` (Burmese structural syllable gate).
// Behaviour is kept identical to the Python reference so that beam search
// pruning matches the official decoder.
// Modified 2026: translated from Python to JavaScript for browser execution.

const CONS = 'က-အ';
const INDEP = 'ဣ-ဪ';
const SYMBOL = '၌၍၎၏';
const VIRAMA = '\u1039';
const ASAT = '\u103A';
const MEDIALS = 'ျြွှ';
const TONES = '့း';
const MARKS = 'ါာိီုူေဲံ့းျြွှ်္';
const DIGITS = '၀၁၂၃၄၅၆၇၈၉';
const EXTRA_CONS = 'ဿၐ-ၙ';
const ONSET = `${CONS}${INDEP}${SYMBOL}${EXTRA_CONS}`;

const BREAK = new RegExp(`(?<!${VIRAMA})(?<!${ASAT}${VIRAMA})([${ONSET}])(?![${ASAT}${VIRAMA}])`, 'u');
const BURMESE = new RegExp(`[${ONSET}ါ-ှ${DIGITS}]`, 'u');
const POLICED = new RegExp(`[${ONSET}ါ-ှ]`, 'u');

const DOUBLE = new RegExp(`([${MARKS}])\\1`, 'u');
const PILEUP = /[ုူ]{2,}|[ိီ]{2,}|[ာါ]{2,}|[ေဲ]{2,}/u;
const ASAT_OK = new RegExp(`(?:[${CONS}${EXTRA_CONS}${INDEP}][${MEDIALS}]*|[ာါ])[${TONES}]?${ASAT}`, 'u');
const MEDIAL_ORDER = /^(?:ျ?ြ?ွ?ှ?)$/u;
const NO_NUCLEUS = new RegExp(`^[${ONSET}][${MEDIALS}]*${ASAT}[${TONES}]?$`, 'u');
const AFTER_ASAT = new RegExp(`${ASAT}[${ONSET}]`, 'u');
const ONSET_RE = new RegExp(`[${ONSET}]`, 'u');

const MAX_SYL = 10;
const WORD_RESET = new Set([' ', '\t', '\n', '\r', '\xa0', '\u200b', '။', '၊']);
const MEDIAL_SET = new Set([...MEDIALS]);

const cpLength = (s) => [...s].length;

function countChar(s, ch) {
  let n = 0;
  for (let i = 0; i < s.length; i++) if (s[i] === ch) n++;
  return n;
}

function marksOk(s) {
  if (DOUBLE.test(s) || PILEUP.test(s) || AFTER_ASAT.test(s)) return false;
  if (countChar(s, ASAT) > 2) return false;
  let med = '';
  for (const c of s) if (MEDIAL_SET.has(c)) med += c;
  if (!MEDIAL_ORDER.test(med)) return false;
  return true;
}

function asatOk(s) {
  for (let i = 0; i < s.length; i++) {
    if (s[i] !== ASAT) continue;
    if (!ASAT_OK.test(s.slice(0, i + 1))) return false;
  }
  return true;
}

export function syllableOk(s, wordInitial) {
  if (!s || cpLength(s) > MAX_SYL) return false;
  if (!ONSET_RE.test(s[0])) return false;
  if (s.endsWith(VIRAMA)) return false;
  if (!marksOk(s) || !asatOk(s)) return false;
  if (wordInitial && NO_NUCLEUS.test(s)) return false;
  return true;
}

// --- state encoding -------------------------------------------------------
// A constraint state is a set of (buffer, wordInitial) pairs. We keep it as a
// canonical string so it can be used directly as a Map key.
export function encodeState(pairs) {
  const parts = pairs.map(([b, w]) => (w ? '1' : '0') + b);
  parts.sort();
  return parts.join('\u0001');
}

function decodeState(key) {
  if (!key) return [];
  return key.split('\u0001').map((p) => [p.slice(1), p[0] === '1']);
}

export class SyllableConstraint {
  constructor() {
    this.complete = null;
    this.prefixes = null;
    this.feasibleCache = new Map();
    this.doneCache = new Map();
    this.stepCache = new Map();
    this.finishCache = new Map();
  }

  startState() {
    return encodeState([['', true]]);
  }

  step(stateKey, ch) {
    const cacheKey = stateKey + '\u0002' + ch;
    const hit = this.stepCache.get(cacheKey);
    if (hit !== undefined) return hit;

    let out;
    const states = decodeState(stateKey);
    if (!POLICED.test(ch)) {
      if (!states.some(([b, w]) => b === '' || this.done(b, w))) {
        out = null;
      } else {
        out = encodeState([['', WORD_RESET.has(ch)]]);
      }
    } else {
      const nxt = [];
      const seen = new Set();
      const add = (b, w) => {
        const k = (w ? '1' : '0') + b;
        if (seen.has(k)) return;
        seen.add(k);
        nxt.push([b, w]);
      };
      for (const [b, w] of states) {
        const grown = b + ch;
        if (this.feasible(grown, w)) add(grown, w);
        const nw = b === '' ? w : false;
        if ((b === '' || this.done(b, w)) && this.feasible(ch, nw)) add(ch, nw);
      }
      out = nxt.length ? encodeState(nxt) : null;
    }

    if (this.stepCache.size > 200000) this.stepCache.clear();
    this.stepCache.set(cacheKey, out);
    return out;
  }

  canFinish(stateKey) {
    const hit = this.finishCache.get(stateKey);
    if (hit !== undefined) return hit;
    const out = decodeState(stateKey).some(([b, w]) => b === '' || this.done(b, w));
    if (this.finishCache.size > 100000) this.finishCache.clear();
    this.finishCache.set(stateKey, out);
    return out;
  }

  feasible(s, wordInitial) {
    const key = (wordInitial ? '1' : '0') + s;
    const hit = this.feasibleCache.get(key);
    if (hit !== undefined) return hit;
    let out = true;
    if (cpLength(s) > MAX_SYL || !marksOk(s) || !asatOk(s)) out = false;
    else if (!ONSET_RE.test(s[0])) out = false;
    else if (this.prefixes !== null && !this.prefixes.has(s)) out = false;
    if (this.feasibleCache.size > 200000) this.feasibleCache.clear();
    this.feasibleCache.set(key, out);
    return out;
  }

  done(s, wordInitial) {
    const key = (wordInitial ? '1' : '0') + s;
    const hit = this.doneCache.get(key);
    if (hit !== undefined) return hit;
    let out = syllableOk(s, wordInitial);
    if (out && this.complete !== null && !this.complete.has(s)) out = false;
    if (this.doneCache.size > 200000) this.doneCache.clear();
    this.doneCache.set(key, out);
    return out;
  }

  valid(text) {
    let st = this.startState();
    for (const ch of text) {
      st = this.step(st, ch);
      if (st === null) return false;
    }
    return this.canFinish(st);
  }
}

export function segment(text) {
  const idx = [];
  for (const m of text.matchAll(new RegExp(BREAK.source, 'gu'))) idx.push(m.index);
  if (!idx.length) return text ? [text] : [];
  const out = [];
  for (let i = 0; i < idx.length; i++) {
    const s = idx[i];
    const e = i + 1 < idx.length ? idx[i + 1] : text.length;
    if (i === 0 && s > 0) out.push(text.slice(0, s));
    out.push(text.slice(s, e));
  }
  return out.filter(Boolean);
}

export function burmeseRuns(line) {
  const runs = [];
  let cur = '';
  for (const ch of line) {
    if (BURMESE.test(ch)) cur += ch;
    else if (cur) {
      runs.push(cur);
      cur = '';
    }
  }
  if (cur) runs.push(cur);
  return runs;
}
