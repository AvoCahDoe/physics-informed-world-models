/** Headless smoke test across all three routes: fails on console errors and
 *  checks that each page's key content actually mounted. */
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

// Anchored to this file, not the cwd, so screenshots land in viz/ whether the
// test is run from here or from the repo root.
const HERE = dirname(fileURLToPath(import.meta.url));
const shot = (name) => join(HERE, name);

const ROOT = process.env.SMOKE_URL ?? "http://localhost:4173";
const errors = [];
const checks = [];
const check = (name, ok, detail = "") => checks.push([name, ok, detail]);

const browser = await puppeteer.launch({
  headless: "new",
  args: ["--no-sandbox", "--enable-unsafe-swiftshader", "--use-gl=swiftshader"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 950, deviceScaleFactor: 1 });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

// Failed asset requests would otherwise be invisible: a 404 on a figure or a
// rollout JSON still renders a "working" page.
const failedRequests = [];
page.on("response", (r) => {
  if (r.status() >= 400) failedRequests.push(`${r.status()} ${r.url()}`);
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// "load" rather than "networkidle0": the render loop keeps the page busy, so
// idle-based waits are unreliable here.
const go = (path) => page.goto(ROOT + path, { waitUntil: "load", timeout: 45000 });

// ---------- / redirects to /docs ----------
await go("/");
await page.waitForSelector(".hero h1", { timeout: 20000 });
check("/ redirects to /docs", new URL(page.url()).pathname === "/docs", page.url());
check("nav has three tabs", (await page.$$(".tab")).length === 3);
check("docs renders model cards", (await page.$$(".card")).length === 3);
await page.screenshot({ path: shot("smoke_docs.png") });

// ---------- /results ----------
await go("/results");
await page.waitForSelector("canvas", { timeout: 20000 });
await page.waitForSelector(".metrics tbody tr", { timeout: 20000 });
await page.waitForSelector(".figure img", { timeout: 20000 });
await sleep(2500);

const results = await page.evaluate(() => ({
  canvas: !!document.querySelector("canvas"),
  charts: document.querySelectorAll(".chart").length,
  paths: document.querySelectorAll(".recharts-line-curve").length,
  chips: document.querySelectorAll(".chip").length,
  metricRows: document.querySelectorAll(".metrics tbody tr").length,
  options: document.querySelectorAll("select option").length,
  figures: document.querySelectorAll(".figure").length,
  time: document.querySelector(".time")?.textContent ?? "",
  errorBox: !!document.querySelector(".error"),
}));

const t1 = results.time;
await sleep(1200);
const t2 = await page.evaluate(() => document.querySelector(".time")?.textContent ?? "");

// Every figure must have decoded, not just have a src attribute.
await page.evaluate(() =>
  Promise.all(
    [...document.querySelectorAll("img")].map((i) => {
      i.loading = "eager";
      return i.decode().catch(() => {});
    }),
  ),
);
await sleep(1500);
const brokenImgs = await page.evaluate(
  () => [...document.querySelectorAll("img")].filter((i) => !i.naturalWidth).length,
);

check("rollout canvas mounted", results.canvas);
check("two charts rendered", results.charts === 2);
check("chart lines drawn", results.paths >= 6, `${results.paths} paths`);
check("series chips present", results.chips === 4);
check("metrics table filled", results.metricRows === 3);
check("trajectory options loaded", results.options === 6);
check("all 13 figures present", results.figures === 13, `${results.figures}`);
check("all figures loaded", brokenImgs === 0, `${brokenImgs} broken`);
check("no data-load error banner", !results.errorBox);
check("clock advancing", t1 !== t2, `${t1} -> ${t2}`);
await page.screenshot({ path: shot("smoke_results.png") });

// The naive-contact failure mode is the app's key demo.
const clicked = await page.evaluate(() => {
  const btn = [...document.querySelectorAll("button")].find(
    (b) => b.textContent?.trim() === "Naive",
  );
  btn?.click();
  return !!btn;
});
check("naive mode toggles", clicked);
if (clicked) {
  await sleep(2200);
  await page.screenshot({ path: shot("smoke_naive.png") });
}

// ---------- /try ----------
await go("/try");
await page.waitForSelector(".ball-canvas", { timeout: 20000 });
await sleep(1500);

const readout = () =>
  page.evaluate(
    () => document.querySelector(".readout b")?.textContent ?? "",
  );
const h1 = await readout();
await sleep(1200);
const h2 = await readout();

check("live sim canvases mounted", (await page.$$(".try-stage canvas")).length === 2);
check("live sim is animating", h1 !== h2, `${h1} -> ${h2}`);
check("controls rendered", (await page.$$(".slider")).length >= 7);
check("presets rendered", (await page.$$(".preset")).length === 5);

// Switching to the elastic preset must actually change the physics.
await page.evaluate(() => {
  const b = [...document.querySelectorAll(".preset")].find(
    (x) => x.textContent?.trim() === "Elastic",
  );
  b?.click();
});
await sleep(600);
const elasticNote = await page.evaluate(
  () => document.querySelector(".note.good")?.textContent ?? "",
);
check("elastic preset applies", elasticNote.includes("exactly conserved"));
await page.screenshot({ path: shot("smoke_try.png") });

await browser.close();

check("no failed requests", failedRequests.length === 0, failedRequests.join(", "));
check("no console errors", errors.length === 0);

let failed = 0;
for (const [name, ok, detail] of checks) {
  if (!ok) failed++;
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}${detail ? `  (${detail})` : ""}`);
}
if (errors.length) console.log("console errors:\n" + errors.join("\n"));
console.log(`\n${checks.length - failed}/${checks.length} checks passed`);
process.exit(failed ? 1 : 0);
