const CACHE_PREFIX = 'nocm';
const SHELL_CACHE = `${CACHE_PREFIX}-shell-v1`;
const STATIC_CACHE = `${CACHE_PREFIX}-static-v1`;
const DICT_CACHE_PREFIX = `${CACHE_PREFIX}-dict`;

const scopeUrl = new URL(self.registration.scope);
const BASE_PATH = scopeUrl.pathname.endsWith('/') ? scopeUrl.pathname : `${scopeUrl.pathname}/`;
const SHELL_URLS = [BASE_PATH, `${BASE_PATH}index.html`];
const STATIC_URLS = [
  `${BASE_PATH}manifest.webmanifest`,
  `${BASE_PATH}icons/icon-192.png`,
  `${BASE_PATH}icons/icon-512.png`,
  `${BASE_PATH}icons/icon-maskable-192.png`,
  `${BASE_PATH}icons/icon-maskable-512.png`,
];

const EXTERNAL_CACHE_HOSTS = new Set([
  'fonts.googleapis.com',
  'fonts.gstatic.com',
  'cdnjs.cloudflare.com',
]);

async function putIfCacheable(cacheName, request, response) {
  if (response && (response.ok || response.type === 'opaque')) {
    const cache = await caches.open(cacheName);
    await cache.put(request, response.clone());
  }
  return response;
}

async function staleWhileRevalidate(request) {
  const cache = await caches.open(STATIC_CACHE);
  const cached = await cache.match(request);
  const networkPromise = fetch(request)
    .then(response => putIfCacheable(STATIC_CACHE, request, response))
    .catch(() => null);

  if (cached) {
    networkPromise.catch(() => {});
    return cached;
  }

  const fresh = await networkPromise;
  return fresh || Response.error();
}

async function networkFirstShell(request) {
  try {
    const response = await fetch(request, { cache: 'no-cache' });
    await putIfCacheable(SHELL_CACHE, request, response);
    return response;
  } catch (_error) {
    return (await caches.match(request)) || (await caches.match(`${BASE_PATH}index.html`));
  }
}

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  await putIfCacheable(STATIC_CACHE, request, response);
  return response;
}

self.addEventListener('install', event => {
  event.waitUntil(
    Promise.all([
      caches.open(SHELL_CACHE).then(cache => cache.addAll(SHELL_URLS)),
      caches.open(STATIC_CACHE).then(cache => cache.addAll(STATIC_URLS)),
    ]).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  const keep = new Set([SHELL_CACHE, STATIC_CACHE]);
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(
        keys
          .filter(key => !keep.has(key) && !key.startsWith(DICT_CACHE_PREFIX))
          .map(key => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);

  if (url.pathname.endsWith('.json.gz')) return;

  if (request.mode === 'navigate') {
    event.respondWith(networkFirstShell(request));
    return;
  }

  if (url.origin !== self.location.origin) {
    if (EXTERNAL_CACHE_HOSTS.has(url.hostname)) {
      event.respondWith(staleWhileRevalidate(request));
    }
    return;
  }

  event.respondWith(cacheFirst(request));
});
