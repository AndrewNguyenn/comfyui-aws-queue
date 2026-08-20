// Verifies the viewer's IndexedDB history cache against a stubbed /jobs API:
//   1 cold load pages through the whole history and snapshots it
//   2 a warm start hydrates from the snapshot in ONE request
//   3 returning to page 0 no longer re-paginates
//   4 a job that completed since the snapshot still reaches the gallery
//   5 the header-count control forces a full re-pagination
//   6 REGRESSION (C1): a head refresh DURING the background load must not
//     persist a truncated history
// puppeteer-core is not a repo dependency (the viewer ships as plain files with
// no build step). Resolve it from wherever it is installed: NODE_PATH, a
// PUPPETEER_DIR override, or a global npm root.
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');
function loadPuppeteer() {
  const tries = [];
  if (process.env.PUPPETEER_DIR) tries.push(process.env.PUPPETEER_DIR);
  try { tries.push(path.join(execSync('npm root -g').toString().trim(), 'puppeteer-core')); }
  catch (_e) { /* npm not on PATH */ }
  tries.push('puppeteer-core');
  for (const t of tries) {
    try { return require(t); } catch (_e) { /* next */ }
  }
  console.error(
    'puppeteer-core not found. Install it and re-run, e.g.:\n' +
    '  npm install -g puppeteer-core && node test-cache.js\n' +
    '  # or: PUPPETEER_DIR=/path/to/node_modules/puppeteer-core node test-cache.js');
  process.exit(2);
}
const puppeteer = loadPuppeteer();

// Build an instrumented copy of the viewer. The two hooks below expose module
// internals the cache tests need to drive; they live here rather than in the
// shipped app.js so production carries no test-only surface.
const APP = path.join(__dirname, '..', 'app.js');
const APP_TEST = path.join(__dirname, 'app.test.js');
{
  const src = fs.readFileSync(APP, 'utf8');
  const anchor = '  /* ---------- init ---------- */';
  if (!src.includes(anchor)) {
    console.error('harness: init anchor not found in app.js — update the test');
    process.exit(1);
  }
  fs.writeFileSync(APP_TEST, src.replace(anchor,
    '  window.__forceHeadRefresh = () => _refreshHead();\n' +
    '  window.__histLoading = () => histLoading;\n\n' + anchor));
}

const CHROME = process.env.CHROME_PATH ||
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PAGE_URL = 'file://' + path.join(__dirname, 'harness.html');
const TOTAL = 19000;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
function check(name, cond, detail) {
  if (!cond) failures++;
  console.log(`  [${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' — ' + detail : ''}`);
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    protocolTimeout: 240000,
    args: ['--no-sandbox', '--allow-file-access-from-files'],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(120000);
  page.on('pageerror', (e) => { console.log('  !! page error:', e.message); failures++; });

  // Read the cached history length. -1 = store absent, 0 = key absent.
  const readCache = () => page.evaluate(() => new Promise((res) => {
    const req = indexedDB.open('viewer.history');
    req.onsuccess = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('jobs')) { db.close(); return res(-1); }
      try {
        const g = db.transaction('jobs', 'readonly').objectStore('jobs').get('history');
        g.onsuccess = () => { const v = g.result; db.close(); res(v && v.jobs ? v.jobs.length : 0); };
        g.onerror = () => { db.close(); res(-2); };
      } catch (_e) { db.close(); res(-3); }
    };
    req.onerror = () => res(-4);
  }));
  const clearCache = () => page.evaluate(() => new Promise((res) => {
    const req = indexedDB.open('viewer.history');
    req.onsuccess = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('jobs')) { db.close(); return res(); }
      const d = db.transaction('jobs', 'readwrite').objectStore('jobs').delete('history');
      d.onsuccess = () => { db.close(); res(); };
      d.onerror = () => { db.close(); res(); };
    };
    req.onerror = () => res();
  }));
  const reqs = () => page.evaluate(() => window.__countJobReqs());
  const items = async () => parseInt(await page.evaluate(
    () => document.getElementById('ct-all').textContent), 10);
  const settle = async (want, tries = 120) => {
    for (let i = 0; i < tries; i++) {
      if ((await readCache()) === want) return true;
      await sleep(500);
    }
    return false;
  };

  // A reliable cold reset. Clearing the cache in-place races two writers: an
  // in-flight _bgLoadRest that will write the full history when it finishes,
  // and _persistHistory's 5s debounce timer left armed by an earlier test.
  // Reloading first tears down both (a fresh document, fresh timers), THEN we
  // clear, then verify it actually stuck before the measured load.
  const coldReset = async () => {
    await page.evaluate(() => window.__setFetchDelay(0));
    await page.reload({ waitUntil: 'domcontentloaded' });
    await sleep(300);
    for (let i = 0; i < 10; i++) {
      await page.reload({ waitUntil: 'domcontentloaded' });  // kill armed timers
      await clearCache();
      await sleep(200);
      if ((await readCache()) <= 0) return true;
    }
    return false;
  };

  // ---------- 1. COLD LOAD ----------
  console.log('\n1. Cold load (cache cleared) — expect a full page-through');
  await page.goto(PAGE_URL, { waitUntil: 'domcontentloaded' });
  await page.evaluate(() => sessionStorage.clear());
  const wasReset = await coldReset();
  check('cache actually cleared before the cold measurement', wasReset, 'reset failed');
  await page.evaluate(() => window.__resetLog());
  await page.reload({ waitUntil: 'domcontentloaded' });
  const coldOk = await settle(TOTAL);
  const coldReqs = await reqs();
  const cachedCount = await readCache();
  console.log(`  cold /jobs requests: ${coldReqs}, cached: ${cachedCount}`);
  check('cold load pages through the history', coldReqs >= 15, `${coldReqs} requests`);
  check('cache holds the complete history', coldOk && cachedCount === TOTAL, `${cachedCount} jobs`);

  // ---------- 2. WARM START ----------
  console.log('\n2. Warm start (cache present) — expect ONE request');
  await page.reload({ waitUntil: 'domcontentloaded' });
  await sleep(2500);
  const warmReqs = await reqs();
  const warmItems = await items();
  console.log(`  warm /jobs requests: ${warmReqs}, items: ${warmItems}`);
  check('warm start issues exactly 1 history request', warmReqs === 1, `${warmReqs} requests`);
  check('warm start shows the full history', warmItems === TOTAL, `ct-all=${warmItems}`);
  check('warm start is a >=15x request reduction', coldReqs / Math.max(warmReqs, 1) >= 15,
        `${coldReqs} -> ${warmReqs}`);

  // ---------- 3. PAGE-0 RETURN ----------
  console.log('\n3. Page 3 -> page 0 — expect no re-pagination');
  await page.evaluate(() => window.__resetLog());
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('#pager .pg-num')];
    (b.find((x) => x.dataset.pg === '3') || b[b.length - 1]).click();
  });
  await sleep(500);
  await page.evaluate(() => {
    const b = [...document.querySelectorAll('#pager .pg-num')];
    const t = b.find((x) => x.dataset.pg === '0');
    if (t) t.click();
  });
  await sleep(2000);
  const navReqs = await reqs();
  console.log(`  page-0 return /jobs requests: ${navReqs}`);
  check('return to page 0 does not re-paginate', navReqs <= 1, `${navReqs} requests`);

  // ---------- 4. NEW JOB REACHES A WARM GALLERY ----------
  console.log('\n4. A job completed since the snapshot still shows up');
  await page.evaluate(() => window.__injectNewJob());
  await page.reload({ waitUntil: 'domcontentloaded' });
  await sleep(2500);
  const afterItems = await items();
  const afterReqs = await reqs();
  console.log(`  items: ${TOTAL} -> ${afterItems} in ${afterReqs} request(s)`);
  check('new job merged on warm start', afterItems === TOTAL + 1, `ct-all=${afterItems}`);
  check('and it only cost one request', afterReqs === 1, `${afterReqs} requests`);

  // ---------- 5. FORCED RELOAD ----------
  console.log('\n5. Header-count control forces a full re-pagination');
  await page.evaluate(() => window.__resetLog());
  await page.evaluate(() => document.getElementById('hdr-count').click());
  await sleep(1000);
  await settle(TOTAL + 1);
  const forcedReqs = await reqs();
  console.log(`  forced reload /jobs requests: ${forcedReqs}`);
  check('forced reload re-paginates from the API', forcedReqs >= 15, `${forcedReqs} requests`);

  // ---------- 6. REGRESSION C1: no truncated snapshot ----------
  console.log('\n6. C1 — head refresh during the background load must not persist a partial history');
  const wasReset6 = await coldReset();
  check('cache actually cleared before the C1 probe', wasReset6, 'reset failed');
  // 120ms per request stretches the ~20-request background load to ~2.4s,
  // which is the window a real 19k-row load occupies. Without the delay the
  // stub finishes before a mid-load probe can run at all.
  await page.evaluate(() => window.__setFetchDelay(120));
  await page.reload({ waitUntil: 'domcontentloaded' });
  // The invariant is "never a PARTIAL snapshot". A read of 0/-1 (nothing
  // written yet) is fine, and so is a read of the full count — the background
  // load can legitimately complete between our histLoading probe and our cache
  // read, since each is a separate round-trip. What must never appear is a
  // count strictly between the two: that is a truncated history, which a warm
  // start would then hydrate and keep forever.
  // _refreshHead returns early when it merges nothing, so simply calling it
  // mid-load never reaches the persist. To reproduce C1 the refresh has to
  // actually ADD a job — so inject one into the stub during the load window,
  // which is exactly the real trigger (a generation completing while the
  // background loader is still paging the tail).
  const FULL = TOTAL + 2; // TOTAL + persistent injection + 1 transient
  let sawLoading = false, injected = false;
  const truncated = [];
  for (let i = 0; i < 30; i++) {
    const loading = await page.evaluate(() => window.__histLoading && window.__histLoading());
    if (loading) {
      sawLoading = true;
      if (!injected) {
        await page.evaluate(() => window.__injectTransient());
        injected = true;
      }
      await page.evaluate(() => window.__forceHeadRefresh && window.__forceHeadRefresh());
      const mid = await readCache();
      if (mid > 0 && mid < FULL) truncated.push(mid);
    }
    await sleep(120);
  }
  check('a job was injected during the load window', injected, 'never saw the window');
  console.log(`  observed background load in progress: ${sawLoading}`);
  console.log(`  truncated snapshots seen: ${truncated.length ? truncated.join(', ') : 'none'}`);
  check('the mid-load window was actually observed', sawLoading, 'guard would be untested otherwise');
  check('no truncated snapshot written mid-load', truncated.length === 0,
        truncated.length ? `saw partial writes of ${truncated.join(', ')}` : 'none');

  await page.evaluate(() => window.__setFetchDelay(0));
  const finalOk = await settle(FULL);
  const finalCount = await readCache();
  console.log(`  cache after the load completed: ${finalCount}`);
  check('final snapshot is the complete history', finalOk && finalCount === FULL,
        `${finalCount} jobs, want ${FULL}`);

  await browser.close();
  console.log(failures === 0 ? '\nALL CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error('harness error:', e); process.exit(1); });
