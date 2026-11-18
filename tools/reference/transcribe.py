# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Nara contributors
"""Local Nara SAR inference: bounded audio, original final decoder, no downloads."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf


def read_audio(path):
    path = Path(path)
    if path.stat().st_size > 256 * 1024 ** 2:
        raise ValueError('Audio exceeds 256 MiB')
    info = sf.info(path)
    if not 1 <= info.channels <= 8 or not 8000 <= info.samplerate <= 192000:
        raise ValueError('Expected 1–8 channels and an 8–192 kHz sample rate')
    if not .25 <= info.frames / info.samplerate <= 1800:
        raise ValueError('Audio must be between 0.25 seconds and 30 minutes')
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError('Audio contains non-finite samples')
    mono = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    if rate != 16000:
        positions = np.arange(len(mono) * 16000 // rate, dtype=np.float64) * (rate / 16000)
        low = positions.astype(np.int64)
        fraction = positions - low
        mono = (mono[low] * (1 - fraction) + mono[np.minimum(low + 1, len(mono) - 1)] * fraction).astype(np.float32)
    return mono


def chunks(samples):
    start = 0
    while start < len(samples):
        end = min(start + 160000, len(samples))
        if end < len(samples):
            limit, quietest = end, math.inf
            for at in range(limit - 32000, limit - 3200 + 1, 320):
                window = samples[at:at + 3200].astype(np.float64)
                energy = float(np.sum(window * window))
                if energy < quietest:
                    quietest = energy
                    end = max(start + 128000, at + 1600)
        if len(samples) - end < 4000:
            end = len(samples)
        yield start, end
        start = end


def normalize(samples):
    values = samples.astype(np.float64)
    values -= values.mean()
    return (values / math.sqrt(float(np.mean(values * values)) + 1e-5)).astype(np.float32)


def top_rows(logits):
    # Stable top-k preserves increasing token ID at ties, matching the app.
    for row in logits:
        ids = np.argsort(-row, kind='stable')[:10]
        yield [(int(index), float(row[index])) for index in ids]


def transcribe(audio_path, model_dir, backend='onnx', precision='int8', threads=2):
    if not 1 <= threads <= 32:
        raise ValueError('Thread count must be between 1 and 32')
    started = time.perf_counter()
    folder = Path(model_dir).resolve()
    metadata = json.loads((folder / 'sar.json').read_text(encoding='utf-8'))
    expected = {'checkpoint': 'run3-step_6000', 'sampleRate': 16000, 'blankId': 0,
                'vocabSize': 9865, 'beam': 16, 'topk': 10, 'alpha': .6, 'beta': 0}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('This runner requires the matching Nara SAR model and final decoder config')
    samples = read_audio(audio_path)
    if backend == 'onnx':
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        name = 'acoustic.onnx' if precision == 'int8' else 'acoustic-fp32.onnx'
        session = ort.InferenceSession(str(folder / 'onnx' / name), options,
                                       providers=['CPUExecutionProvider'])
        def infer(values):
            return session.run(None, {'audio': values[None]})[0][0]
    else:
        import torch
        from sar_acoustic import SarAcoustic
        precision = 'fp32'
        torch.set_num_threads(threads)
        model = SarAcoustic(folder / 'model.safetensors').eval()
        def infer(values):
            with torch.inference_mode():
                return model(torch.from_numpy(values[None]))[0].numpy()
    from sar_decoder import Decoder
    decoder = Decoder(folder)
    parts = []
    completed = 0
    try:
        for start, end in chunks(samples):
            values = normalize(samples[start:end])
            if len(values) > 168000:
                raise ValueError('Audio chunk exceeds model bounds')
            if np.any(np.abs(values) > 1e-5):
                logits = infer(values)
                if logits.ndim != 2 or logits.shape[1] != 9865 or not np.isfinite(logits).all():
                    raise ValueError('Model returned invalid acoustic output')
                text = decoder.decode(top_rows(logits)).strip()
                if text:
                    parts.append(text)
            completed += 1
    finally:
        decoder.close()
    return {'text': ' '.join(parts), 'audio_seconds': len(samples) / 16000,
            'chunks': completed, 'backend': backend, 'precision': precision,
            'elapsed_seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--backend', choices=['onnx', 'torch'], default='onnx')
    parser.add_argument('--precision', choices=['int8', 'fp32'], default='int8')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    try:
        result = transcribe(args.audio, args.model_dir, args.backend, args.precision, args.threads)
    except Exception as error:
        parser.exit(1, 'Nara SAR: ' + str(error) + '\n')
    print(json.dumps(result, ensure_ascii=False) if args.json else result['text'])


if __name__ == '__main__':
    main()
