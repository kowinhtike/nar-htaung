# SPDX-License-Identifier: Apache-2.0
"""Dump a Python reference run (Nara SAR) so the JS port can be checked against it.

Usage:
  python tools/dump_reference.py <audio.wav> <out-dir>

Writes mono.f32 (16 kHz float32 samples), chunk{i}.f32 (normalised chunk input),
chunk{i}.logits.f32 (frames x 9865 float32 logits) and meta.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'reference'))

import transcribe as T  # noqa: E402
from sar_decoder import Decoder  # noqa: E402


def main():
    audio = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    model = ROOT / 'models'
    sar = json.loads((model / 'sar.json').read_text(encoding='utf-8'))

    samples = T.read_audio(audio)
    samples.astype('<f4').tofile(out / 'mono.f32')

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model / 'acoustic.onnx'), options,
                                   providers=['CPUExecutionProvider'])

    decoder = Decoder(model)
    parts = []
    meta = {'chunks': [], 'text': '', 'audio_seconds': len(samples) / 16000}
    try:
        for index, (start, end) in enumerate(T.chunks(samples)):
            values = T.normalize(samples[start:end])
            values.astype('<f4').tofile(out / f'chunk{index}.f32')
            logits = session.run(None, {'audio': values[None]})[0][0]
            logits.astype('<f4').tofile(out / f'chunk{index}.logits.f32')
            text = decoder.decode(T.top_rows(logits)).strip()
            meta['chunks'].append({
                'start': int(start), 'end': int(end),
                'frames': int(logits.shape[0]), 'vocab': int(logits.shape[1]),
                'text': text, 'peak': float(np.abs(values).max()),
            })
            if text:
                parts.append(text)
    finally:
        decoder.close()

    meta['text'] = ' '.join(parts)
    meta['vocab_size'] = int(sar['vocabSize'])
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
