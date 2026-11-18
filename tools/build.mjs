// SPDX-License-Identifier: Apache-2.0
// Assemble the static deployment bundle in dist/.
//
// Only the app shell ships: the 430 MB of model weights are never uploaded.
// The client fetches them from Hugging Face on first load and caches them in
// the browser (see src/worker.js), which also keeps every file well under
// Cloudflare Pages' 25 MiB per-file limit.

import { cpSync, mkdirSync, readdirSync, rmSync, statSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const dist = join(root, 'dist');

const FILES = ['index.html', 'styles.css', '_headers'];
const DIRECTORIES = ['src', 'vendor'];

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });

for (const file of FILES) {
  const from = join(root, file);
  if (!existsSync(from)) throw new Error(`missing ${file}`);
  cpSync(from, join(dist, file));
}

for (const directory of DIRECTORIES) {
  const from = join(root, directory);
  if (!existsSync(from)) throw new Error(`missing ${directory}/`);
  cpSync(from, join(dist, directory), { recursive: true });
}

let total = 0;
let count = 0;
let largest = { path: '', size: 0 };
const walk = (path) => {
  const info = statSync(path);
  if (info.isDirectory()) {
    for (const entry of readdirSync(path)) walk(join(path, entry));
    return;
  }
  total += info.size;
  count += 1;
  if (info.size > largest.size) largest = { path, size: info.size };
};
walk(dist);

const limit = 25 * 1024 * 1024;
console.log(`dist/ ready: ${count} files, ${(total / 1e6).toFixed(1)} MB`);
console.log(`largest: ${largest.path.slice(root.length + 1)} (${(largest.size / 1e6).toFixed(1)} MB)`);
if (largest.size > limit) {
  console.error('largest file exceeds the Cloudflare Pages 25 MiB per-asset limit');
  process.exit(1);
}
