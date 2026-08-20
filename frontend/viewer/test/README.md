# viewer cache tests

Regression tests for the gallery's IndexedDB history cache
(`_histCache` / `_persistHistory` / `loadJobs` in `../app.js`).

    npm install puppeteer-core     # once, anywhere on PATH
    node test-cache.js

`harness.html` stands up the minimum DOM `app.js` queries plus a stubbed,
request-counting `/jobs` API serving 19,000 keyset-paginated jobs, so the tests
measure real request volume without touching AWS. The runner generates
`app.test.js` (gitignored) — `app.js` plus two hooks — so the shipped file
carries no test-only surface.

What it pins down:

| # | Behaviour |
|---|---|
| 1 | A cold load pages through the whole history and snapshots it |
| 2 | A warm start hydrates from the snapshot in **one** request, not ~20 |
| 3 | Returning to page 0 no longer re-paginates |
| 4 | A job that completed since the snapshot still reaches the gallery |
| 5 | The header-count control forces a full re-pagination |
| 6 | **C1 regression** — a head refresh *during* the background load must not persist a truncated history |

Test 6 is the important one. `_bgLoadRest` fills `allItems` incrementally, so
while it runs the list is a PREFIX of the history. `_refreshHead` (fired by the
pending poll whenever a generation completes) and `doDelete` both call
`_persistHistory`, and either can land in that window. Without the `histLoading`
guard the cache is overwritten with the prefix — and because a warm start never
re-paginates, the next boot hydrates the truncation and keeps it. Removing the
guard makes test 6 fail with a snapshot of ~1,150 jobs instead of ~19,002.

Note the test must INJECT a job mid-load to reproduce it: `_refreshHead` returns
early when it merges nothing, so it never reaches the persist call otherwise.
An earlier version of this test missed the bug for exactly that reason.
