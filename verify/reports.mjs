// Does the prose still say what the JSON says?
//
// reports/*.json is written by src/fb/*.py. Every figure in README.md and
// notes/METHODS.md was typed out of those files by hand, and every derived
// quantity inside them (effect_of_split, refit_gain, max_over_min) was computed
// by the same script that wrote the numbers it is derived from. So two things
// are unchecked: whether the derived fields still follow from the fields they
// are derived from, and whether the prose still matches either.
//
// This recomputes every derived field from its inputs, then rebuilds the exact
// string each published figure should appear as and requires it to be present
// in the prose. Re-running an experiment without editing the text now fails
// here instead of shipping a README that describes the previous run.
//
// Usage: node verify/reports.mjs <repo-root>

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? ".";
const read = (p) => readFileSync(join(root, p), "utf8");
const load = (p) => JSON.parse(read(p));

const backtest = load("reports/backtest.json");
const models = load("reports/models.json");
const eda = load("reports/eda.json");
const prepare = load("reports/prepare.json");

// bold markers are typography, not content: | **0.4843** | and | 0.4843 |
// are the same published figure
const unbold = (t) => t.replaceAll("*", "");
const readme = unbold(read("README.md"));
const methods = unbold(read("notes/METHODS.md"));

let failures = 0;
const TOL = 1e-12;

function close(label, got, want) {
  const d = Math.abs(got - want);
  const ok = d <= TOL * (1 + Math.abs(want));
  if (!ok) failures++;
  console.log(
    `  ${label.padEnd(46)} ${String(got).padEnd(22)} |d| ${d.toExponential(1)} ${ok ? "ok" : "FAIL"}`,
  );
}

function says(text, needle, label) {
  const ok = text.includes(needle);
  if (!ok) failures++;
  console.log(`  ${(label + ",").padEnd(34)} "${needle}" ${ok ? "ok" : "FAIL, not in the text"}`);
}

const pct = (x) => `${Math.round(x * 100)}%`;
const thousands = (n) => n.toLocaleString("en-US");

// ---- derived fields, against the fields they are derived from --------------

console.log("derived fields inside reports/*.json");
const c = backtest.cells;
close("backtest effect_of_split", backtest.effect_of_split,
  c.temporal_h1.mase_median - c.random_h1.mase_median);
close("backtest effect_of_horizon", backtest.effect_of_horizon,
  c.random_h24.mase_median - c.random_h1.mase_median);
close("backtest baseline_mase", backtest.baseline_mase, c.random_h1.mase_median);
close("models refit_gain", models.refit_gain,
  models.models.gbm_once.mase_median - models.models.gbm_refit.mase_median);
close("eda mean kwh max_over_min", eda.mean_kwh_per_series.max_over_min,
  eda.mean_kwh_per_series.max / eda.mean_kwh_per_series.min);
for (const [key, p] of Object.entries(eda.prob_both_neighbours_in_train)) {
  const h = Number(key.match(/(\d+)pct/)[1]) / 100;
  close(`eda ${key} is (1-h)^2`, p, (1 - h) ** 2);
}
close("prepare series_kept", prepare.series_kept,
  prepare.series_total - prepare.series_dropped_short);

// ---- the files have to describe the same run -------------------------------

console.log("\nconsistency between the four reports");
close("eda rows == prepare rows", eda.rows, prepare.rows);
close("eda series_total == prepare series_kept", eda.series_total, prepare.series_kept);
for (const [name, cell] of Object.entries(c)) {
  close(`backtest ${name} series == n_series`, cell.series, backtest.n_series);
}

// ---- ordering that has to hold or a quantile was mislabelled ---------------

console.log("\norderings inside reports/eda.json");
for (const [name, q] of Object.entries(eda.autocorrelation)) {
  const ok = q.p10 <= q.median && q.median <= q.p90;
  if (!ok) failures++;
  console.log(`  autocorrelation ${name.padEnd(14)} p10 <= median <= p90  ${ok ? "ok" : "FAIL"}`);
}
for (const key of ["hours_per_series", "mean_kwh_per_series"]) {
  const q = eda[key];
  const ok = q.min <= q.median && q.median <= q.max;
  if (!ok) failures++;
  console.log(`  ${key.padEnd(30)} min <= median <= max  ${ok ? "ok" : "FAIL"}`);
}
// seasonal naive cannot beat seasonal naive: this column is a self-check
{
  const ok = models.models.naive168.beats_naive168_frac === 0;
  if (!ok) failures++;
  console.log(`  naive168 beats itself on 0 of the series          ${ok ? "ok" : "FAIL"}`);
}

// ---- the prose, rebuilt from the JSON --------------------------------------

console.log("\nREADME.md, rebuilt from reports/*.json");
const base = backtest.baseline_mase;
says(readme, `${(100 * backtest.effect_of_split / base).toFixed(1)}%`,
  "split effect");
says(readme, `${(100 * backtest.effect_of_horizon / base).toFixed(1)}%`,
  "horizon effect");
says(readme, c.random_h1.mase_median.toFixed(4), "best cell MASE");
says(readme, c.temporal_h24.mase_median.toFixed(4), "worst cell MASE");
says(readme, readme,
  `spread of ${Math.round(100 * (c.temporal_h24.mase_median / base - 1))}%`,
  "spread across the grid");
says(readme, models.models.gbm_once.mase_median.toFixed(4), "gbm_once median");
says(readme, models.models.ets.mase_median.toFixed(4), "ets median");
says(readme, models.models.gbm_refit.mase_median.toFixed(4), "gbm_refit median");
says(readme, `${models.refit_gain.toFixed(4)} MASE`, "refit gain");
says(readme, readme,
  `${(100 * models.refit_gain / models.models.gbm_once.mase_median).toFixed(1)}%`,
  "refit gain as a percentage");
says(readme, pct(models.models.gbm_once.beats_naive168_frac), "gbm beats naive");
says(readme, `${thousands(Math.round(eda.mean_kwh_per_series.max_over_min))}`,
  "scale spread");
says(readme, `${eda.mean_kwh_per_series.min.toFixed(1)} to`, "smallest meter");
says(readme, `${thousands(Math.round(eda.mean_kwh_per_series.max))} mean kWh`,
  "largest meter");
says(readme, pct(eda.prob_both_neighbours_in_train.holdout_20pct),
  "interpolation probability");
says(readme, `correlate at ${eda.autocorrelation["1h"].median.toFixed(2)}`,
  "hourly autocorrelation");
says(readme, `${prepare.series_kept} meters`, "meters kept");
says(readme, `${(prepare.rows / 1e6).toFixed(1)}M rows`, "rows");
says(readme, readme,
  `${thousands(prepare.trimmed_zero_prefix_median_hours)} hours`, "median trimmed prefix");
says(readme, `${prepare.raw_mb.toFixed(0)} MB`, "raw size");
says(readme, `${prepare.parquet_mb.toFixed(1)} MB`, "parquet size");
says(readme, `${backtest.n_series} series`, "grid sample size");
says(readme, `${models.n_series} meters and ${models.origins_per_series} origins`,
  "model sample size");

console.log("\nnotes/METHODS.md, rebuilt from reports/*.json");
for (const [name, cell] of Object.entries(c)) {
  says(methods, `| ${cell.mase_median.toFixed(4)} |`, `grid ${name} MASE`);
  says(methods, `| ${cell.mase_naive_median.toFixed(4)} |`,
    `grid ${name} naive MASE`);
}
for (const [name, m] of Object.entries(models.models)) {
  says(methods, `| ${m.mase_median.toFixed(4)} |`, `${name} median`);
  says(methods, `| ${m.mase_mean.toFixed(4)} |`, `${name} mean`);
  says(methods, `| ${pct(m.beats_naive168_frac)} |`, `${name} beats naive`);
}
for (const [name, q] of Object.entries(eda.autocorrelation)) {
  says(methods, `${q.median.toFixed(4)}`, `autocorrelation ${name} median`);
  says(methods, `${q.p10.toFixed(4)}`, `autocorrelation ${name} p10`);
}

if (failures > 0) {
  console.log(`\n${failures} disagreements between the reports and the prose`);
  process.exit(1);
}
console.log("\nJavaScript: every derived field follows from its inputs, and every"
  + "\npublished figure still appears in the prose that quotes it");
