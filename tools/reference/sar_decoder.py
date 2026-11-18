# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nara contributors
"""Portable Nara SAR final decoding: original beam rules and bilingual ARPA LM.

No Torch, NumPy or KenLM runtime is required. The compact memory-mapped LM is
built from the production ARPA and checked against KenLM before packaging.
"""
import base64
from collections import defaultdict
from functools import lru_cache
import json
import math
import mmap
from pathlib import Path
import struct
from sar_syllables import SyllableConstraint


class LanguageModel:
    def __init__(self, path):
        self.file = open(path, 'rb')
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if self.data[:8] != b'NARALM01':
            raise ValueError('Invalid SAR language model')
        length = struct.unpack_from('<I', self.data, 8)[0]
        self.header = json.loads(self.data[12:12 + length])
        self.start = 12 + length
        self.vocab = {word: index for index, word in enumerate(self.header['vocab'])}
        self.unknown = self.vocab['<unk>']
        self.bos = (self.vocab['<s>'],)
        self.lookup = lru_cache(maxsize=65536)(self._lookup)
        self.advance = lru_cache(maxsize=32768)(self._advance)

    def close(self):
        self.lookup.cache_clear()
        self.advance.cache_clear()
        self.data.close()
        self.file.close()

    def _lookup(self, ids):
        if not ids:
            return None
        order = len(ids)
        block = self.header['blocks'][order - 1]
        width = self.header['keyWidth']
        key = b''.join(index.to_bytes(width, 'big') for index in ids)
        record_size = len(key) + 8
        first = self.start + block['offset']
        low, high = 0, block['count']
        while low < high:
            middle = (low + high) // 2
            at = first + middle * record_size
            candidate = self.data[at:at + len(key)]
            if candidate < key:
                low = middle + 1
            elif candidate > key:
                high = middle
            else:
                return struct.unpack_from('<ff', self.data, at + len(key))
        return None

    def _advance(self, state, char):
        word = '▁' if char == ' ' else char
        index = self.vocab.get(word, self.unknown)
        history = state
        backoff = 0.0
        while True:
            found = self.lookup(history + (index,))
            if found is not None:
                score = backoff + found[0]
                break
            if not history:
                raise ValueError('Missing SAR language-model unigram')
            context = self.lookup(history)
            if context is not None:
                backoff += context[1]
            history = history[1:]
        return score, (state + (index,))[-5:]


class Decoder:
    def __init__(self, folder):
        folder = Path(folder)
        self.metadata = json.loads((folder / 'sar.json').read_text(encoding='utf-8'))
        self.id2char = {int(k): v for k, v in self.metadata['id2char'].items()}
        self.lm = LanguageModel(folder / 'language.lm')
        self.constraint = SyllableConstraint()
        for name in ('_feasible', '_done', 'step', 'can_finish'):
            setattr(self.constraint, name, lru_cache(maxsize=16384)(getattr(self.constraint, name)))

    def close(self):
        self.lm.close()

    def decode(self, rows, cancelled=lambda: False):
        gate = self.constraint
        beams = {'': (0.0, -math.inf, 0.0, self.lm.bos, gate.start_state())}

        def logsum(a, b):
            if a == -math.inf:
                return b
            if b == -math.inf:
                return a
            high = max(a, b)
            return high + math.log(math.exp(a - high) + math.exp(b - high))

        for row in rows:
            if cancelled():
                raise InterruptedError('cancelled')
            candidates = sorted(row, key=lambda item: -item[1])[:10]
            shift = candidates[0][1]
            following = defaultdict(lambda: [-math.inf, -math.inf, 0.0, None, None])
            for prefix, (pb, pnb, lm_score, state, cs) in beams.items():
                total = logsum(pb, pnb)
                last = prefix[-1] if prefix else None
                for index, value in candidates:
                    lp = value - shift  # Common per-frame shift preserves CTC rankings.
                    if index == 0:
                        item = following[prefix]
                        item[0] = logsum(item[0], total + lp)
                        item[2:] = lm_score, state, cs
                        continue
                    char = self.id2char.get(index)
                    if char is None:
                        continue
                    next_cs = gate.step(cs, char)
                    if next_cs is None:
                        continue
                    next_score, next_state = self.lm.advance(state, char)
                    next_prefix = prefix + char
                    if char == last:
                        item = following[prefix]
                        item[1] = logsum(item[1], pnb + lp)
                        item[2:] = lm_score, state, cs
                        probability = pb + lp
                    else:
                        probability = total + lp
                    item = following[next_prefix]
                    item[1] = logsum(item[1], probability)
                    item[2:] = lm_score + next_score, next_state, next_cs
            if not following:
                raise ValueError('SAR final decoder has no legal beam')
            beams = dict(sorted(following.items(),
                key=lambda item: -(logsum(item[1][0], item[1][1]) + .6 * item[1][2]))[:16])
        legal = {text: item for text, item in beams.items() if gate.can_finish(item[4])}
        return max((legal or beams).items(),
            key=lambda item: logsum(item[1][0], item[1][1]) + .6 * item[1][2])[0]

    def decode_base64(self, data, frames):
        values = list(struct.iter_unpack('<if', base64.b64decode(data)))
        if len(values) != frames * 10:
            raise ValueError('Invalid SAR acoustic output')
        return self.decode(values[i:i + 10] for i in range(0, len(values), 10))


_decoder = None
_folder = None


def decode_topk(folder, data, frames):
    global _decoder, _folder
    if _folder != folder:
        if _decoder is not None:
            _decoder.close()
        _decoder = Decoder(folder)
        _folder = folder
    return _decoder.decode_base64(data, frames)


def unload():
    global _decoder, _folder
    if _decoder is not None:
        _decoder.close()
    _decoder = _folder = None
