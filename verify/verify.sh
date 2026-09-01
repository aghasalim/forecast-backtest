#!/usr/bin/env bash
# Recompute the metric this project publishes in every language here, and
# require them to agree.
#
# Every MASE in reports/*.json came out of one function, rolling_backtest() in
# src/fb/harness.py, and every figure and every table downstream reads that same
# output. So nothing in the repository ever checked it. These are independent
# implementations from verify/fixture/history.csv and forecasts.csv, and a
# mistake in the Python would have to be made identically in C, SQL, Go, R and
# JavaScript to survive.
#
# Each is skipped with a clear message if its toolchain is absent, so this runs
# on a laptop with only some of them. CI has all of them.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
tmp="${TMPDIR:-/tmp}"

pass=0 fail=0 skip=0

run () {
    local name="$1" tool="$2"; shift 2
    printf '\n=== %s ===\n' "$name"
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'skipped: %s is not installed\n' "$tool"
        skip=$((skip + 1)); return
    fi
    if "$@"; then pass=$((pass + 1)); else fail=$((fail + 1)); fi
}

# SQL has no assertion of its own, so every row it prints carries its own
# verdict and this counts them. The floor of 50 is there so a query that
# silently returns nothing cannot pass.
check_sql () {
    local out ok bad
    # stdin must be closed: sqlite3 reads it, and inside this script that is
    # the script itself, so without /dev/null it swallows the rest of the file
    # and returns nothing. Its CSV writer also emits CRLF, so strip the \r
    # before anchoring a pattern to end of line.
    out=$(sqlite3 -init verify/mase.sql :memory: "" < /dev/null 2>&1 | tr -d '\r') \
        || { echo "$out"; return 1; }
    ok=$(printf '%s\n' "$out" | grep -c ',ok$')
    bad=$(printf '%s\n' "$out" | grep -c ',FAIL$')
    if [ "$bad" -ne 0 ] || [ "$ok" -lt 50 ]; then
        printf '%s\n' "$out" | grep -v ',ok$' | head -20
        printf 'SQL: %s checks passed, %s failed\n' "$ok" "$bad"
        return 1
    fi
    printf 'SQL reproduces %s fold errors, MASEs and aggregates\n' "$ok"
    return 0
}

check_c () {
    cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror \
       -o "$tmp/fb_mase" verify/mase.c -lm || return 1
    "$tmp/fb_mase" "$root"
}

check_go () { ( cd verify/gocheck && go run . -root "$root" ); }

check_rust () { ( cd verify/foldgrid && cargo run --release --quiet -- "$root" ); }

run "SQL, the aggregation"            sqlite3 check_sql
run "C, the metric kernel"            cc      check_c
run "Go, files and fold leakage"      go      check_go
run "R, the metric and the premise"   Rscript Rscript verify/verify.R "$root"
run "Rust, every layout, simulation"  cargo   check_rust
run "JavaScript, reports against prose" node  node verify/reports.mjs "$root"

printf '\n%s\n' "----------------------------------------"
printf '%d passed, %d failed, %d skipped\n' "$pass" "$fail" "$skip"
[ "$fail" -eq 0 ] || exit 1
[ "$pass" -gt 0 ] || { echo "nothing ran"; exit 1; }
