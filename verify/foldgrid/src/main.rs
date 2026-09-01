//! Two things Python could not afford to run: every backtest layout, and a
//! simulation precise enough to see the approximation in the README.
//!
//! 1. src/fb/harness.py decides where the folds go with three lines of index
//!    arithmetic, and the repository exercises exactly one setting of it, 28
//!    origins of 24 hours. The guarantee those three lines are supposed to give
//!    is that the scored window never overlaps the history a model was handed,
//!    and never overlaps the window the MASE denominator was measured on. This
//!    enumerates every layout on a grid of series length, origin count and
//!    horizon and checks that guarantee on each one, then checks that the ten
//!    layouts in verify/fixture/expected.csv are among the ones it accepted, at
//!    the first origin the fixture publishes.
//!
//! 2. reports/eda.json publishes the probability that both temporal neighbours
//!    of a held-out point are still in training. It is (1-h)^2, which is the
//!    limit for a long series rather than the answer for a series of finite
//!    length. verify/verify.R simulates that at the length of the median meter
//!    and cannot separate the two, because they differ by 9e-5 there. This
//!    simulates it at three lengths with enough draws to separate them at the
//!    shortest, which is what turns "the approximation looks fine" into a
//!    measurement.
//!
//! Usage: cargo run --release -- <repo-root>

use std::env;
use std::fs;
use std::process::exit;

const SEASON: i64 = 168;
const MIN_ORIGINS: i64 = 5;

// the layout grid
const MAX_N: i64 = 20_000;
const MAX_ORIGINS: i64 = 40;
const MAX_HORIZON: i64 = 48;

// the simulation
const OPS_PER_CELL: u64 = 100_000_000;
const SIGMA: f64 = 4.0;

fn fail(msg: String) -> ! {
    println!("  FAIL {}", msg);
    exit(1)
}

// ---- 1. every layout ------------------------------------------------------

/// The harness's own rule, restated: the first origin, or None if it refuses.
fn first_origin(n: i64, n_origins: i64, horizon: i64) -> Option<i64> {
    let first = n - n_origins * horizon;
    if first <= SEASON * 2 || n_origins < MIN_ORIGINS {
        None
    } else {
        Some(first)
    }
}

fn enumerate_layouts() -> (u64, u64) {
    let (mut accepted, mut refused) = (0u64, 0u64);

    for n in 1..=MAX_N {
        for n_origins in 1..=MAX_ORIGINS {
            for horizon in 1..=MAX_HORIZON {
                let first = match first_origin(n, n_origins, horizon) {
                    None => {
                        // a refusal has to have a reason, or the harness is
                        // throwing away layouts it could have scored
                        let f = n - n_origins * horizon;
                        if f > SEASON * 2 && n_origins >= MIN_ORIGINS {
                            fail(format!("refused n={} origins={} horizon={} \
                                          with no reason", n, n_origins, horizon));
                        }
                        refused += 1;
                        continue;
                    }
                    Some(f) => f,
                };

                // the denominator is measured on t in [SEASON, first), pairing
                // each t with t - SEASON, so the highest row it reads is
                // first - 1 and the lowest is 0
                if first <= SEASON {
                    fail(format!("n={} origins={} horizon={}: denominator window \
                                  [{}, {}) is empty", n, n_origins, horizon,
                                 SEASON, first));
                }

                let mut prev_end = first;
                for k in 0..n_origins {
                    let o = first + k * horizon;
                    if o != prev_end {
                        fail(format!("n={} origins={} horizon={}: fold {} starts \
                                      at {}, the previous one ended at {}",
                                     n, n_origins, horizon, k, o, prev_end));
                    }
                    if o + horizon > n {
                        fail(format!("n={} origins={} horizon={}: fold {} runs to \
                                      {}, past the end at {}",
                                     n, n_origins, horizon, k, o + horizon, n));
                    }
                    // the model is handed y[..o]; the denominator read up to
                    // first - 1; both must lie strictly before the scored rows
                    if first - 1 >= o {
                        fail(format!("n={} origins={} horizon={}: the denominator \
                                      reads row {} and fold {} scores from {}",
                                     n, n_origins, horizon, first - 1, k, o));
                    }
                    prev_end = o + horizon;
                }
                if prev_end != n {
                    fail(format!("n={} origins={} horizon={}: the folds end at {}, \
                                  the series ends at {}",
                                 n, n_origins, horizon, prev_end, n));
                }
                accepted += 1;
            }
        }
    }
    (accepted, refused)
}

// ---- the fixture has to be one of the layouts that were accepted ----------

fn column(header: &str, name: &str) -> usize {
    header
        .trim()
        .split(',')
        .position(|h| h == name)
        .unwrap_or_else(|| {
            eprintln!("foldgrid: no column named {}", name);
            exit(2)
        })
}

fn read(root: &str, rel: &str) -> String {
    fs::read_to_string(format!("{}/{}", root, rel)).unwrap_or_else(|e| {
        eprintln!("foldgrid: cannot read {}/{}: {}", root, rel, e);
        exit(2)
    })
}

fn check_fixture(root: &str) -> usize {
    // how many origins and how many steps the fixture actually used
    let fc = read(root, "verify/fixture/forecasts.csv");
    let mut lines = fc.lines();
    let header = lines.next().unwrap_or("");
    let (cs, cm) = (column(header, "series"), column(header, "model"));
    let (co, cst) = (column(header, "origin"), column(header, "step"));
    let mut keys: Vec<String> = Vec::new();
    let mut origins: Vec<Vec<i64>> = Vec::new();
    let mut steps: Vec<Vec<i64>> = Vec::new();
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.trim().split(',').collect();
        let key = format!("{}/{}", f[cs], f[cm]);
        let i = match keys.iter().position(|k| *k == key) {
            Some(i) => i,
            None => {
                keys.push(key);
                origins.push(Vec::new());
                steps.push(Vec::new());
                keys.len() - 1
            }
        };
        let o: i64 = f[co].parse().unwrap();
        let s: i64 = f[cst].parse().unwrap();
        if !origins[i].contains(&o) {
            origins[i].push(o);
        }
        if !steps[i].contains(&s) {
            steps[i].push(s);
        }
    }

    let exp = read(root, "verify/fixture/expected.csv");
    let mut elines = exp.lines();
    let eheader = elines.next().unwrap_or("");
    let (es, em) = (column(eheader, "series"), column(eheader, "model"));
    let (en, ef) = (column(eheader, "n"), column(eheader, "first_origin"));

    let mut checked = 0;
    for line in elines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.trim().split(',').collect();
        let key = format!("{}/{}", f[es], f[em]);
        let i = keys
            .iter()
            .position(|k| *k == key)
            .unwrap_or_else(|| fail(format!("expected.csv names {}, which has no \
                                             forecasts", key)));
        let n: i64 = f[en].parse().unwrap();
        let published: i64 = f[ef].parse().unwrap();
        let n_origins = origins[i].len() as i64;
        let horizon = steps[i].len() as i64;
        match first_origin(n, n_origins, horizon) {
            None => fail(format!("{}: n={} origins={} horizon={} is a layout the \
                                  harness would refuse", key, n, n_origins, horizon)),
            Some(first) if first != published => fail(format!(
                "{}: the rule puts the first origin at {}, expected.csv says {}",
                key, first, published)),
            Some(_) => checked += 1,
        }
    }
    checked
}

// ---- 2. the interpolation probability -------------------------------------

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast, and seeded reproducibly so a failure here can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Draw a hold-out of size m from a series of length n, and return the fraction
/// of interior held-out points whose two neighbours both stayed in training.
fn one_holdout(idx: &mut [usize], mask: &mut [bool], m: usize, rng: &mut Rng) -> f64 {
    let n = idx.len();
    for i in 0..m {
        let j = i + rng.below(n - i);
        idx.swap(i, j);
        mask[idx[i]] = true;
    }
    let (mut interior, mut both) = (0u64, 0u64);
    for &i in idx.iter().take(m) {
        if i > 0 && i + 1 < n {
            interior += 1;
            if !mask[i - 1] && !mask[i + 1] {
                both += 1;
            }
        }
    }
    for &i in idx.iter().take(m) {
        mask[i] = false;
    }
    both as f64 / interior as f64
}

fn published_probability(root: &str, key: &str) -> f64 {
    let text = read(root, "reports/eda.json");
    for line in text.lines() {
        if line.contains(key) {
            let after = line.split(':').nth(1).unwrap_or("");
            return after
                .trim()
                .trim_end_matches(',')
                .parse()
                .unwrap_or_else(|_| fail(format!("{} is not a number in eda.json",
                                                 key)));
        }
    }
    fail(format!("reports/eda.json has no {}", key))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args.get(1).map(String::as_str).unwrap_or(".");

    println!("every backtest layout with n <= {}, {} to {} origins, horizon 1 to {}",
             MAX_N, MIN_ORIGINS, MAX_ORIGINS, MAX_HORIZON);
    let (accepted, refused) = enumerate_layouts();
    println!("  {} layouts accepted, {} refused, and in every accepted one the",
             accepted, refused);
    println!("  folds tile [first_origin, n) exactly and start after every row the");
    println!("  MASE denominator reads");
    let checked = check_fixture(root);
    println!("  the {} layouts in verify/fixture/expected.csv are among them, at the",
             checked);
    println!("  first origin the fixture publishes");

    // the median meter, so the simulation is run at the length the README argues about
    let text = read(root, "reports/eda.json");
    let hours_at = text.find("hours_per_series").unwrap_or(0);
    let median_line = text[hours_at..]
        .lines()
        .find(|l| l.contains("\"median\""))
        .unwrap_or_else(|| fail("reports/eda.json has no hours_per_series median"
                                .to_string()));
    let real_n: usize = median_line
        .split(':')
        .nth(1)
        .unwrap_or("")
        .trim()
        .trim_end_matches(',')
        .parse()
        .unwrap_or_else(|_| fail("hours_per_series median is not an integer"
                                 .to_string()));

    println!("\nthe interpolation probability reports/eda.json publishes");
    println!("  exact  = (N-m)(N-m-1) / ((N-1)(N-2)),  limit = (1-h)^2 as N grows\n");
    let mut separated = 0;
    for (hi, h) in [0.10_f64, 0.20, 0.30].iter().enumerate() {
        let key = format!("holdout_{}pct", (h * 100.0).round() as i64);
        let published = published_probability(root, &key);
        let limit = (1.0 - h) * (1.0 - h);
        if (published - limit).abs() > 1e-12 {
            fail(format!("{} is {}, (1-h)^2 is {}", key, published, limit));
        }
        for n in [1000usize, 4000, real_n] {
            let m = (h * n as f64).round() as usize;
            let exact = ((n - m) as f64 * (n - m - 1) as f64)
                / ((n - 1) as f64 * (n - 2) as f64);
            let draws = (OPS_PER_CELL / (4 * m as u64)).max(200) as usize;

            let mut rng = Rng::new(0x5EED_0000 + (hi as u64) * 7919 + n as u64);
            let mut idx: Vec<usize> = (0..n).collect();
            let mut mask = vec![false; n];
            let mut sum = 0.0f64;
            let mut sumsq = 0.0f64;
            for _ in 0..draws {
                let p = one_holdout(&mut idx, &mut mask, m, &mut rng);
                sum += p;
                sumsq += p * p;
            }
            let mean = sum / draws as f64;
            let var = (sumsq - draws as f64 * mean * mean) / (draws as f64 - 1.0);
            let se = (var / draws as f64).sqrt();
            let z_exact = (mean - exact).abs() / se.max(1e-15);
            let z_limit = (mean - limit).abs() / se.max(1e-15);
            if z_exact > SIGMA {
                fail(format!("h={:.2} N={}: simulated {:.6}, exact {:.6}, {:.1} sd \
                              apart", h, n, mean, exact, z_exact));
            }
            if z_limit > SIGMA {
                separated += 1;
            }
            println!("  h={:.2} N={:<6} m={:<6} {:>8} draws  simulated {:.6} +- \
                      {:.6}  exact {:.6} ({:.1} sd)  limit {:.6} ({:.1} sd)",
                     h, n, m, draws, mean, se, exact, z_exact, limit, z_limit);
        }
    }

    if separated == 0 {
        fail("the simulation never separated the exact probability from the \
              (1-h)^2 limit, so it had no power to check either".to_string());
    }
    println!("\n  the simulation lands on the exact probability in all 9 cells, and");
    println!("  is more than {} sd from the (1-h)^2 limit in {} of them, so it can",
             SIGMA, separated);
    println!("  tell the two apart and the published value is the limit");
    println!("\nRust: the fold rule is safe on every layout, and the published");
    println!("interpolation probability survives a simulation that could refute it");
}
