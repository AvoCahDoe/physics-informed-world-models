/** Headless smoke test: load the built app, fail on console errors, and check
 *  that the scene, charts and controls actually mounted. */
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer";

// Anchored to this file, not the cwd, so screenshots land in viz/ whether the
// test is run from here or from the repo root.
const HERE = dirname(fileURLToPath(import.meta.url));
const shot = (name) => join(HERE, name);

const URL = process.env.SMOKE_URL ?? "http://localhost:4173/";
const errors = [];

const browser = await puppeteer.launch({
  headless: "new",
  args: ["--no-sandbox", "--enable-unsafe-swiftshader", "--use-gl=swiftshader"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 950, deviceScaleFactor: 1 });
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));

// "load" rather than "networkidle0": the render loop keeps the page busy, so
// idle-based waits are unreliable here.
await page.goto(URL, { waitUntil: "load", timeout: 45000 });
await page.waitForSelector("canvas", { timeout: 20000 });
await page.waitForSelector(".metrics tbody tr", { timeout: 20000 });
await new Promise((r) => setTimeout(r, 2500));

const report = await page.evaluate(() => ({
  canvas: !!document.querySelector("canvas"),
  charts: document.querySelectorAll(".chart").length,
  paths: document.querySelectorAll(".recharts-line-curve").length,
  chips: document.querySelectorAll(".chip").length,
  metricRows: document.querySelectorAll(".metrics tbody tr").length,
  options: document.querySelectorAll("select option").length,
  time: document.querySelector(".time")?.textContent ?? "",
  errorBox: !!document.querySelector(".error"),
}));

// Confirm the shared clock is actually advancing.
const t1 = report.time;
await new Promise((r) => setTimeout(r, 1200));
const t2 = await page.evaluate(() => document.querySelector(".time")?.textContent ?? "");

await page.screenshot({ path: shot("smoke.png"), fullPage: false });

// Also capture the naive-contact failure mode, which is the app's key demo.
const clicked = await page.evaluate(() => {
  const btn = [...document.querySelectorAll("button")].find(
    (b) => b.textContent?.trim() === "Naive",
  );
  btn?.click();
  return !!btn;
});
if (clicked) {
  await new Promise((r) => setTimeout(r, 2200));
  await page.screenshot({ path: shot("smoke_naive.png"), fullPage: false });
}
await browser.close();

const checks = [
  ["canvas mounted", report.canvas],
  ["two charts rendered", report.charts === 2],
  ["chart lines drawn", report.paths >= 6],
  ["series chips present", report.chips === 4],
  ["metrics table filled", report.metricRows === 3],
  ["trajectory options loaded", report.options === 6],
  ["no data-load error banner", !report.errorBox],
  ["clock advancing", t1 !== t2],
  ["no console errors", errors.length === 0],
];

let failed = 0;
for (const [name, ok] of checks) {
  if (!ok) failed++;
  console.log(`[${ok ? "PASS" : "FAIL"}] ${name}`);
}
if (errors.length) console.log("console errors:\n" + errors.join("\n"));
console.log(JSON.stringify({ ...report, t1, t2 }));
process.exit(failed ? 1 : 0);
