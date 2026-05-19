// Minimal service worker: cache the app shell so the PWA opens offline.
// Identification still needs network; this just lets the page open instantly
// and gives "Add to home screen" / installability.
const CACHE = 'mixid-shell-v1';
const SHELL = ['/', '/manifest.webmanifest'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // Never cache API calls — always go to network.
  if (url.pathname.startsWith('/jobs') || url.pathname === '/stats') return;
  // Shell pages: cache-first
  if (e.request.mode === 'navigate' || SHELL.includes(url.pathname)) {
    e.respondWith(
      caches.match(e.request).then(c => c || fetch(e.request).then(r => {
        const copy = r.clone();
        caches.open(CACHE).then(cache => cache.put(e.request, copy));
        return r;
      }).catch(() => caches.match('/')))
    );
  }
});
