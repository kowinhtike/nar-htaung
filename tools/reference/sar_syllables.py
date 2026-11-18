# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nara contributors
import argparse
import json
import re
import unicodedata
CONS = 'က-အ'
INDEP = 'ဣ-ဪ'
SYMBOL = '၌၍၎၏'
VIRAMA = '္'
ASAT = '်'
MEDIALS = 'ျြွှ'
TONES = '့း'
MARKS = 'ါာိီုူေဲံ့းျြွှ်္'
DIGITS = '၀၁၂၃၄၅၆၇၈၉'
EXTRA_CONS = 'ဿၐ-ၙ'
ONSET = f'{CONS}{INDEP}{SYMBOL}{EXTRA_CONS}'
BREAK = re.compile(f'(?<!{VIRAMA})(?<!{ASAT}{VIRAMA})([{ONSET}])(?![{ASAT}{VIRAMA}])')
BURMESE = re.compile(f'[{ONSET}ါ-ှ{DIGITS}]')
POLICED = re.compile(f'[{ONSET}ါ-ှ]')

def nfc(t):
    return unicodedata.normalize('NFC', t or '')

def segment(text):
    idx = [m.start() for m in BREAK.finditer(text)]
    if not idx:
        return [text] if text else []
    out = []
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(text)
        if i == 0 and s > 0:
            out.append(text[:s])
        out.append(text[s:e])
    return [s for s in out if s]

def burmese_runs(line):
    cur = []
    for ch in line:
        if BURMESE.match(ch):
            cur.append(ch)
        elif cur:
            yield ''.join(cur)
            cur = []
    if cur:
        yield ''.join(cur)
_DOUBLE = re.compile(f'([{MARKS}])\\1')
_PILEUP = re.compile('[ုူ]{2,}|[ိီ]{2,}|[ာါ]{2,}|[ေဲ]{2,}')
_ASAT_OK = re.compile(f'(?:[{CONS}{EXTRA_CONS}{INDEP}][{MEDIALS}]*|[ာါ])[{TONES}]?{ASAT}')
_MEDIAL_ORDER = re.compile('ျ?ြ?ွ?ှ?$')
_NO_NUCLEUS = re.compile(f'^[{ONSET}][{MEDIALS}]*{ASAT}[{TONES}]?$')
_AFTER_ASAT = re.compile(f'{ASAT}[{ONSET}]')
MAX_SYL = 10
WORD_RESET = set(' \t\n\r\xa0\u200b။၊')

def _marks_ok(s):
    if _DOUBLE.search(s) or _PILEUP.search(s) or _AFTER_ASAT.search(s):
        return False
    if s.count(ASAT) > 2:
        return False
    med = ''.join((c for c in s if c in MEDIALS))
    if not _MEDIAL_ORDER.fullmatch(med):
        return False
    return True

def _asat_ok(s, final):
    for m in re.finditer(ASAT, s):
        i = m.start()
        if not _ASAT_OK.search(s[:i + 1]):
            return False
    return True
_ONSET_RE = re.compile(f'[{ONSET}]')

def syllable_ok(s, word_initial):
    if not s or len(s) > MAX_SYL:
        return False
    if not _ONSET_RE.match(s[0]):
        return False
    if s.endswith(VIRAMA):
        return False
    if not _marks_ok(s) or not _asat_ok(s, True):
        return False
    if word_initial and _NO_NUCLEUS.match(s):
        return False
    return True

class SyllableConstraint:

    def __init__(self, syllables=None):
        self.complete = None
        self.prefixes = None
        if syllables:
            keep = {nfc(s) for s in syllables}
            keep = {s for s in keep if len(s) <= MAX_SYL and (not any((c in DIGITS for c in s))) and syllable_ok(s, word_initial=False)}
            canon, alt = ('့်', '့်')
            keep |= {s.replace(canon, alt) for s in keep if canon in s}
            self.complete = keep
            self.prefixes = set()
            for s in keep:
                for i in range(1, len(s) + 1):
                    self.prefixes.add(s[:i])

    def start_state(self):
        return frozenset({('', True)})

    def step(self, states, ch):
        if not POLICED.match(ch):
            if not any((b == '' or self._done(b, w) for b, w in states)):
                return None
            return frozenset({('', ch in WORD_RESET)})
        nxt = set()
        for b, w in states:
            grown = b + ch
            if self._feasible(grown, w):
                nxt.add((grown, w))
            nw = w if b == '' else False
            if (b == '' or self._done(b, w)) and self._feasible(ch, nw):
                nxt.add((ch, nw))
        return frozenset(nxt) if nxt else None

    def can_finish(self, states):
        return any((b == '' or self._done(b, w) for b, w in states))

    def _feasible(self, s, word_initial):
        if len(s) > MAX_SYL or not _marks_ok(s) or (not _asat_ok(s, False)):
            return False
        if not _ONSET_RE.match(s[0]):
            return False
        if self.prefixes is not None and s not in self.prefixes:
            return False
        return True

    def _done(self, s, word_initial):
        if not syllable_ok(s, word_initial):
            return False
        if self.complete is not None and s not in self.complete:
            return False
        return True

    def valid(self, text):
        st = self.start_state()
        for ch in nfc(text):
            st = self.step(st, ch)
            if st is None:
                return False
        return self.can_finish(st)
