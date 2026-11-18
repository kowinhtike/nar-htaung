// SPDX-License-Identifier: Apache-2.0
// Minimal static server for the standalone client.
//
// Cross-origin isolation headers (COOP/COEP) are required for
// SharedArrayBuffer, which is what lets ONNX Runtime Web use more than one
// WASM thread. Large model files are marked immutable so the browser keeps
// them on disk after the first download.

import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index !== -1 && args[index + 1] ? args[index + 1] : fallback;
};

const here = resolve(fileURLToPath(new URL('.', import.meta.url)));
// --root dist serves the production bundle built by `npm run build`.
const root = resolve(here, flag('root', process.env.ROOT || '.'));
const port = Number(flag('port', process.env.PORT || 4173));

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.lm': 'application/octet-stream',
  '.wav': 'audio/wav',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

function headersFor(pathname) {
  const immutable = pathname.startsWith('/models/') || pathname.startsWith('/vendor/');
  return {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',
    'Cache-Control': immutable ? 'public, max-age=31536000, immutable' : 'no-cache',
    'Access-Control-Allow-Origin': '*',
  };
}

const server = createServer((request, response) => {
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname.endsWith('/')) pathname += 'index.html';

  const target = join(root, normalize(pathname).replace(/^(\.\.[/\\])+/, ''));
  if (!target.startsWith(root)) {
    response.writeHead(403).end('Forbidden');
    return;
  }

  let info;
  try {
    info = statSync(target);
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
    return;
  }
  if (info.isDirectory()) {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' }).end('Not found');
    return;
  }

  const headers = {
    ...headersFor(pathname),
    'Content-Type': types[extname(target).toLowerCase()] || 'application/octet-stream',
    'Accept-Ranges': 'bytes',
  };

  const range = request.headers.range;
  if (range) {
    const match = /bytes=(\d*)-(\d*)/.exec(range);
    if (match) {
      const start = match[1] ? Number(match[1]) : 0;
      const end = match[2] ? Math.min(Number(match[2]), info.size - 1) : info.size - 1;
      if (start <= end) {
        response.writeHead(206, {
          ...headers,
          'Content-Range': `bytes ${start}-${end}/${info.size}`,
          'Content-Length': end - start + 1,
        });
        createReadStream(target, { start, end }).pipe(response);
        return;
      }
    }
  }

  response.writeHead(200, { ...headers, 'Content-Length': info.size });
  if (request.method === 'HEAD') {
    response.end();
    return;
  }
  createReadStream(target).pipe(response);
});

server.listen(port, () => {
  console.log(`NarHtaung: http://localhost:${port}`);
  console.log('Cross-origin isolation is enabled, so WASM threads are available.');
});
