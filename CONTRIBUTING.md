# Contributing

Thanks for taking a look. This project is deliberately small: plain ES modules, no bundler, no framework, one runtime dependency.

## Setup

```bash
npm install
npm start                      # http://localhost:4173
```

Model weights are not committed. Either run `bash tools/fetch-model.sh` (~430 MB) or let the app fetch them from Hugging Face on first load.

## Ground rules

1. **Do not change the decoding recipe.** Beam 16, top-k 10, α 0.6, β 0, blank id 0, the language model container and the Burmese syllable rules must stay identical to the Python reference. Anything else produces different text and defeats the point of the port.
2. **Keep the port readable.** `src/decoder.js`, `src/lm.js`, `src/syllables.js` and `src/audio.js` should stay traceable to their Python counterparts. If you deviate, say why in a comment.
3. **No new runtime dependencies** unless there is no reasonable alternative. The whole point is a zero-build client.
4. **Keep new source files Apache-2.0 headed.** Use the `SPDX-License-Identifier: Apache-2.0` tag at the top.

## Before opening a PR

If you touched anything in `src/decoder.js`, `src/lm.js`, `src/syllables.js` or `src/audio.js`, run the parity harness and paste the summary into the PR:

```bash
bash tools/make-sample.sh                                    # macOS `say`, or supply your own WAV
python tools/dump_reference.py samples/sample-en-long.wav /tmp/ref
node tools/parity.mjs /tmp/ref                               # expect "ALL CHECKS PASSED"
```

`tools/dump_reference.py` needs `numpy`, `onnxruntime` and `soundfile`; `tools/reference/` holds the unmodified Python decoder used as the oracle.

Also run:

```bash
npm run check
```

## Good first issues

- Safari/Firefox verification and a browser support table with real numbers.
- A service worker so the app shell works offline too (weights are already cached in OPFS).
- Word-level timestamps from CTC alignment (the model itself emits none).
- A `?sample=` demo path so the README screenshot can be reproduced without a local file.
