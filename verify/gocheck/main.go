// Structural validation of every results file, and a proof that no backtest
// fold reads data at or after its own forecast origin.
//
// The claim the whole repository rests on is in src/fb/harness.py: "a model
// cannot look at the future because it was never given it". That is enforced
// by handing over y[:origin], and it is checked by a Python self-check that
// runs in the same process as the thing it is checking. This checks it from the
// outside, on the recorded output: for every fold, the value the forecaster
// produced must be a history value at an index strictly before the origin, and
// the folds must tile the evaluation window exactly.
//
// It also walks reports/*.json and verify/fixture/*.csv looking for the damage
// a truncated write or a drifted column does: ragged rows, duplicate or empty
// column names, NaN, Inf, null, and row counts that disagree with each other.
// Nothing else in the repository looks at those files as files.
//
// Usage: go run . -root ..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

const (
	season = 168
	tol    = 1e-12
)

var problems int

func fail(format string, a ...any) {
	fmt.Printf("  FAIL "+format+"\n", a...)
	problems++
}

func die(format string, a ...any) {
	fmt.Fprintf(os.Stderr, "gocheck: "+format+"\n", a...)
	os.Exit(2)
}

// ---- files ----------------------------------------------------------------

type table struct {
	path   string
	header []string
	rows   [][]string
	index  map[string]int
}

func readCSV(path string) *table {
	f, err := os.Open(path)
	if err != nil {
		die("cannot open %s: %v", path, err)
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		die("%s: %v", path, err)
	}
	if len(rows) < 2 {
		die("%s: only %d rows", path, len(rows))
	}
	t := &table{path: path, header: rows[0], rows: rows[1:], index: map[string]int{}}
	for i, h := range t.header {
		t.index[h] = i
	}
	return t
}

func (t *table) col(name string) int {
	i, ok := t.index[name]
	if !ok {
		die("%s has no column %q", t.path, name)
	}
	return i
}

func (t *table) num(row []string, name string) float64 {
	v, err := strconv.ParseFloat(strings.TrimSpace(row[t.col(name)]), 64)
	if err != nil {
		die("%s: %q in column %s is not a number", t.path, row[t.col(name)], name)
	}
	return v
}

func (t *table) intAt(row []string, name string) int {
	return int(t.num(row, name))
}

// validate reports every structural problem in one file rather than the first,
// so a broken run is diagnosed in one pass.
func validate(t *table) {
	seen := map[string]bool{}
	for _, h := range t.header {
		if strings.TrimSpace(h) == "" {
			fail("%s: a column has an empty name", filepath.Base(t.path))
		}
		if seen[h] {
			fail("%s: duplicate column %q", filepath.Base(t.path), h)
		}
		seen[h] = true
	}
	for i, row := range t.rows {
		for j, cell := range row {
			low := strings.ToLower(strings.TrimSpace(cell))
			if low == "" || low == "nan" || low == "inf" || low == "-inf" ||
				low == "infinity" || low == "-infinity" {
				fail("%s row %d column %s is %q",
					filepath.Base(t.path), i+2, t.header[j], cell)
			}
		}
	}
}

// walkJSON refuses the values a division can leak into a report: NaN, Inf and
// null. Go's decoder already refuses NaN and Inf as bare tokens, so this
// catches the string spellings pandas and numpy write.
func walkJSON(path string, v any, where string) {
	switch x := v.(type) {
	case map[string]any:
		for k, sub := range x {
			walkJSON(path, sub, where+"."+k)
		}
	case []any:
		for i, sub := range x {
			walkJSON(path, sub, fmt.Sprintf("%s[%d]", where, i))
		}
	case float64:
		if math.IsNaN(x) || math.IsInf(x, 0) {
			fail("%s: %s is %v", filepath.Base(path), where, x)
		}
	case string:
		low := strings.ToLower(strings.TrimSpace(x))
		if low == "nan" || low == "inf" || low == "-inf" || low == "infinity" {
			fail("%s: %s is the string %q", filepath.Base(path), where, x)
		}
	case nil:
		fail("%s: %s is null", filepath.Base(path), where)
	}
}

// ---- the fixture ----------------------------------------------------------

type pair struct {
	series, model string
	origins       []int
	yhat          map[int][]float64
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	// every tracked results file, as a file
	jsons, _ := filepath.Glob(filepath.Join(*root, "reports", "*.json"))
	csvs, _ := filepath.Glob(filepath.Join(*root, "verify", "fixture", "*.csv"))
	sort.Strings(jsons)
	sort.Strings(csvs)
	if len(jsons) == 0 || len(csvs) == 0 {
		die("found %d reports/*.json and %d fixture CSVs", len(jsons), len(csvs))
	}

	fmt.Printf("validating %d JSON reports and %d fixture CSVs\n",
		len(jsons), len(csvs))
	before := problems
	for _, p := range jsons {
		b, err := os.ReadFile(p)
		if err != nil {
			die("cannot read %s: %v", p, err)
		}
		var v any
		if err := json.Unmarshal(b, &v); err != nil {
			fail("%s: %v", filepath.Base(p), err)
			continue
		}
		walkJSON(p, v, filepath.Base(p))
	}
	tables := map[string]*table{}
	for _, p := range csvs {
		t := readCSV(p)
		tables[filepath.Base(p)] = t
		validate(t)
	}
	if problems == before {
		fmt.Println("  well formed, no ragged rows, duplicate columns, empty" +
			" cells, NaN, Inf or null")
	}

	// load the fixture
	hist := map[string][]float64{}
	ht := tables["history.csv"]
	for _, row := range ht.rows {
		s := row[ht.col("series")]
		if ht.intAt(row, "t") != len(hist[s]) {
			fail("history.csv: series %s is not in t order at t=%d", s,
				ht.intAt(row, "t"))
		}
		hist[s] = append(hist[s], ht.num(row, "y"))
	}

	pairs := map[string]*pair{}
	order := []string{}
	ft := tables["forecasts.csv"]
	horizon := 0
	for _, row := range ft.rows {
		key := row[ft.col("series")] + "/" + row[ft.col("model")]
		p, ok := pairs[key]
		if !ok {
			p = &pair{series: row[ft.col("series")], model: row[ft.col("model")],
				yhat: map[int][]float64{}}
			pairs[key] = p
			order = append(order, key)
		}
		o := ft.intAt(row, "origin")
		step := ft.intAt(row, "step")
		if _, ok := p.yhat[o]; !ok {
			p.origins = append(p.origins, o)
		}
		for len(p.yhat[o]) <= step {
			p.yhat[o] = append(p.yhat[o], math.NaN())
		}
		if !math.IsNaN(p.yhat[o][step]) {
			fail("forecasts.csv: %s origin %d step %d appears twice", key, o, step)
		}
		p.yhat[o][step] = ft.num(row, "yhat")
		if step+1 > horizon {
			horizon = step + 1
		}
	}
	sort.Strings(order)
	for _, k := range order {
		sort.Ints(pairs[k].origins)
	}

	// the row counts have to agree with each other, or one of these files was
	// written by a different run than the others
	fmt.Printf("\n%d series, %d series-model pairs, horizon %d\n",
		len(hist), len(pairs), horizon)
	wantForecastRows := 0
	wantFoldRows := 0
	for _, k := range order {
		wantForecastRows += len(pairs[k].origins) * horizon
		wantFoldRows += len(pairs[k].origins)
	}
	if len(ft.rows) != wantForecastRows {
		fail("forecasts.csv has %d rows, the origins and horizon imply %d",
			len(ft.rows), wantForecastRows)
	}
	if got := len(tables["folds.csv"].rows); got != wantFoldRows {
		fail("folds.csv has %d rows, there are %d folds", got, wantFoldRows)
	}
	if got := len(tables["expected.csv"].rows); got != len(pairs) {
		fail("expected.csv has %d rows, there are %d series-model pairs",
			got, len(pairs))
	}

	// ---- the part that matters: no fold reads at or after its origin -------
	//
	// Both fixture models are lag rules, so the index each forecast came from
	// is known exactly. Checking the index rather than the value is what makes
	// this a statement about leakage: a forecast is only clean if the row it
	// copied lies strictly before the origin.
	lag := map[string]int{"seasonal_naive": season, "daily_naive": 24}

	fmt.Printf("\nfold boundaries and leakage, %d folds\n", wantFoldRows)
	checkedSteps := 0
	for _, k := range order {
		p := pairs[k]
		y := hist[p.series]
		n := len(y)
		first := n - len(p.origins)*horizon

		if first <= 2*season {
			fail("%s: first origin %d is not more than 2*%d hours in", k, first, season)
		}
		for i, o := range p.origins {
			if want := first + i*horizon; o != want {
				fail("%s: fold %d starts at %d, the arithmetic says %d", k, i, o, want)
			}
		}
		if last := p.origins[len(p.origins)-1] + horizon; last != n {
			fail("%s: the last fold ends at %d, the series ends at %d", k, last, n)
		}
		// the MASE denominator window must end before the first origin, so the
		// yardstick never overlaps the window being scored
		if first > n {
			fail("%s: denominator window [%d,%d) runs past the series", k, season, first)
		}

		l, known := lag[p.model]
		if !known {
			fail("%s: no lag rule for model %s, cannot check leakage", k, p.model)
			continue
		}
		for _, o := range p.origins {
			for step, v := range p.yhat[o] {
				src := o - l + step
				if src >= o {
					fail("%s origin %d step %d would read index %d, which is not"+
						" before the origin", k, o, step, src)
					continue
				}
				if src < 0 || src >= n {
					fail("%s origin %d step %d reads index %d, outside the series",
						k, o, step, src)
					continue
				}
				if math.Abs(v-y[src]) > 1e-9 {
					fail("%s origin %d step %d is %.9g, y[%d] before the origin"+
						" is %.9g", k, o, step, v, src, y[src])
					continue
				}
				checkedSteps++
			}
		}
	}
	fmt.Printf("  %d of %d forecast values traced to a history index strictly"+
		" before their origin\n", checkedSteps, wantForecastRows)
	fmt.Println("  folds tile the evaluation window exactly and end at the last hour")

	// ---- one recompute: MASE, from the definition --------------------------
	fmt.Println("\nMASE, against verify/fixture/expected.csv")
	et := tables["expected.csv"]
	worst := 0.0
	for _, row := range et.rows {
		key := row[et.col("series")] + "/" + row[et.col("model")]
		p, ok := pairs[key]
		if !ok {
			fail("expected.csv names %s, which has no forecasts", key)
			continue
		}
		y := hist[p.series]
		first := len(y) - len(p.origins)*horizon
		if got := et.intAt(row, "first_origin"); got != first {
			fail("%s: expected.csv says first_origin %d, the arithmetic says %d",
				key, got, first)
		}
		if got := et.intAt(row, "n"); got != len(y) {
			fail("%s: expected.csv says n=%d, history.csv has %d", key, got, len(y))
		}
		denom := 0.0
		for j := season; j < first; j++ {
			denom += math.Abs(y[j] - y[j-season])
		}
		denom /= float64(first - season)
		total := 0.0
		for _, o := range p.origins {
			mae := 0.0
			for step, v := range p.yhat[o] {
				mae += math.Abs(y[o+step] - v)
			}
			total += mae / float64(horizon)
		}
		got := total / float64(len(p.origins)) / denom
		want := et.num(row, "mase")
		d := math.Abs(got-want) / (1 + math.Abs(want))
		if d > worst {
			worst = d
		}
		if d > tol {
			fail("%s: Go %.15g, published %.15g", key, got, want)
		}
	}
	fmt.Printf("  %d MASEs recomputed, worst relative gap %.1e\n", len(et.rows), worst)

	if problems > 0 {
		fmt.Printf("\n%d problems\n", problems)
		os.Exit(1)
	}
	fmt.Println("\nGo: results files are well formed, no fold sees its own future," +
		" and MASE agrees")
}
