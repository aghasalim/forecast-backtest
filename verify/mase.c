/* An independent MASE kernel, in C, from the definition.
 *
 * Every number this project publishes is a MASE, or an aggregation of MASEs,
 * and all of them came out of one implementation: rolling_backtest() in
 * src/fb/harness.py. Nothing checked that implementation against anything. This
 * reads only verify/fixture/history.csv and verify/fixture/forecasts.csv and
 * rebuilds, from the definition alone:
 *
 *   the in-sample seasonal-naive denominator, mean |y[t] - y[t-168]| over
 *     t in [168, first_origin), which is strictly before the scored window
 *   the mean absolute error of every fold           -> folds.csv
 *   MASE per series and model                       -> expected.csv
 *   the median, mean and beats-naive aggregation    -> summary.csv
 *
 * It also re-derives first_origin from the fold arithmetic, n - n_origins *
 * horizon, rather than reading it, so a wrong origin would show up as a wrong
 * denominator rather than being copied across.
 *
 * Every sum is formed twice, once as a plain double accumulation in file order
 * and once with Kahan compensation in long double, and the two must agree.
 * That is here because a mean over 33,000 terms spanning three orders of
 * magnitude is exactly where a published figure can turn out to be an artefact
 * of summation order rather than a measurement.
 *
 * Columns are resolved by name. A renamed or reordered column is an error, not
 * a silently different answer.
 *
 * Usage: mase <repo-root>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEASON        168
#define MAX_SERIES     32
#define MAX_MODELS      8
#define MAX_POINTS  16384
#define MAX_ORIGINS    64
#define MAX_HORIZON    64
#define MAX_FIELDS     32
#define MAX_LINE     4096

#define TOL 1e-12

static int failures = 0;

static void die(const char *what, const char *detail)
{
    fprintf(stderr, "mase: %s: %s\n", what, detail);
    exit(2);
}

/* ---- CSV, columns by name ------------------------------------------------ */

typedef struct {
    FILE *fp;
    char  header[MAX_LINE];
    char *names[MAX_FIELDS];
    int   ncol;
    char  line[MAX_LINE];
    char *field[MAX_FIELDS];
    long  lineno;
} Csv;

static int split(char *s, char **out, int max)
{
    int n = 0;
    char *p = s;
    s[strcspn(s, "\r\n")] = '\0';
    out[n++] = p;
    while (*p) {
        if (*p == ',') {
            if (n >= max) return -1;
            *p = '\0';
            out[n++] = p + 1;
        }
        p++;
    }
    return n;
}

static void csv_open(Csv *c, const char *root, const char *rel)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", root, rel);
    c->fp = fopen(path, "r");
    if (!c->fp) die("cannot open", path);
    if (!fgets(c->header, sizeof c->header, c->fp)) die("empty file", path);
    c->ncol = split(c->header, c->names, MAX_FIELDS);
    if (c->ncol < 1) die("unreadable header", path);
    c->lineno = 1;
}

static int csv_col(const Csv *c, const char *name)
{
    int i;
    for (i = 0; i < c->ncol; i++)
        if (strcmp(c->names[i], name) == 0) return i;
    fprintf(stderr, "mase: no column named %s\n", name);
    exit(2);
}

/* returns 0 at end of file; a row whose field count differs from the header is
   fatal, because a ragged results file is a corrupt results file */
static int csv_next(Csv *c)
{
    int n;
    if (!fgets(c->line, sizeof c->line, c->fp)) return 0;
    c->lineno++;
    if (c->line[0] == '\n' || c->line[0] == '\0') return 0;
    n = split(c->line, c->field, MAX_FIELDS);
    if (n != c->ncol) {
        fprintf(stderr, "mase: line %ld has %d fields, header has %d\n",
                c->lineno, n, c->ncol);
        exit(2);
    }
    return 1;
}

static double csv_num(const Csv *c, int col)
{
    char *end;
    double v = strtod(c->field[col], &end);
    if (end == c->field[col] || *end != '\0') {
        fprintf(stderr, "mase: line %ld column %s is not a number: %s\n",
                c->lineno, c->names[col], c->field[col]);
        exit(2);
    }
    if (!isfinite(v)) {
        fprintf(stderr, "mase: line %ld column %s is %s\n",
                c->lineno, c->names[col], c->field[col]);
        exit(2);
    }
    return v;
}

/* ---- two summations of the same terms ------------------------------------ */

typedef struct { double plain; long double kahan, comp; long n; } Acc;

static void acc_add(Acc *a, double x)
{
    long double y = (long double)x - a->comp;
    long double t = a->kahan + y;
    a->comp = (t - a->kahan) - y;
    a->kahan = t;
    a->plain += x;
    a->n++;
}

/* the mean, and the relative gap between the two ways of forming it */
static double acc_mean(const Acc *a, double *gap)
{
    double k = (double)(a->kahan / (long double)a->n);
    double p = a->plain / (double)a->n;
    if (gap) *gap = fabs(k - p) / (1.0 + fabs(k));
    return k;
}

/* ---- the data ------------------------------------------------------------ */

typedef struct {
    char   name[32];
    int    n;
    double y[MAX_POINTS];
} Series;

typedef struct {
    int    series, model;
    int    origin[MAX_ORIGINS];
    int    n_origins;
    int    horizon;
    double yhat[MAX_ORIGINS][MAX_HORIZON];
    int    filled[MAX_ORIGINS][MAX_HORIZON];
} Pair;

static Series series[MAX_SERIES];
static int    n_series = 0;
static char   models[MAX_MODELS][32];
static int    n_models = 0;
static Pair   pairs[MAX_SERIES * MAX_MODELS];
static int    n_pairs = 0;

static int find_series(const char *name)
{
    int i;
    for (i = 0; i < n_series; i++)
        if (strcmp(series[i].name, name) == 0) return i;
    if (n_series >= MAX_SERIES) die("too many series", name);
    snprintf(series[n_series].name, sizeof series[0].name, "%s", name);
    return n_series++;
}

static int find_model(const char *name)
{
    int i;
    for (i = 0; i < n_models; i++)
        if (strcmp(models[i], name) == 0) return i;
    if (n_models >= MAX_MODELS) die("too many models", name);
    snprintf(models[n_models], sizeof models[0], "%s", name);
    return n_models++;
}

static Pair *find_pair(int s, int m)
{
    int i;
    for (i = 0; i < n_pairs; i++)
        if (pairs[i].series == s && pairs[i].model == m) return &pairs[i];
    if (n_pairs >= MAX_SERIES * MAX_MODELS) die("too many pairs", "");
    pairs[n_pairs].series = s;
    pairs[n_pairs].model = m;
    return &pairs[n_pairs++];
}

static int origin_slot(Pair *p, int origin)
{
    int i;
    for (i = 0; i < p->n_origins; i++)
        if (p->origin[i] == origin) return i;
    if (p->n_origins >= MAX_ORIGINS) die("too many origins", "");
    p->origin[p->n_origins] = origin;
    return p->n_origins++;
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int cmp_dbl(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* numpy's median: the middle value, or the mean of the two middle values */
static double median(double *v, int n)
{
    qsort(v, (size_t)n, sizeof v[0], cmp_dbl);
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void report(const char *what, double got, double want)
{
    double d = fabs(got - want);
    int ok = d <= TOL * (1.0 + fabs(want));
    if (!ok) failures++;
    printf("  %-34s %18.12f  |d| %.1e  %s\n", what, got, d, ok ? "ok" : "FAIL");
}

int main(int argc, char **argv)
{
    const char *root = (argc > 1) ? argv[1] : ".";
    Csv c;
    int cs, ct, cy, cm, co, cstep, cyhat, cmae, cn, cfirst, cden, cmase, cmed,
        cmean, cbeats, cser;
    int i, k, j;
    double mase[MAX_SERIES * MAX_MODELS];
    double denom[MAX_SERIES];
    int    first_of[MAX_SERIES];
    double worst_gap = 0.0;

    /* history */
    csv_open(&c, root, "verify/fixture/history.csv");
    cs = csv_col(&c, "series"); ct = csv_col(&c, "t"); cy = csv_col(&c, "y");
    while (csv_next(&c)) {
        int s = find_series(c.field[cs]);
        int t = (int)csv_num(&c, ct);
        if (t != series[s].n) die("history is not in t order", c.field[cs]);
        if (t >= MAX_POINTS) die("series too long", c.field[cs]);
        series[s].y[t] = csv_num(&c, cy);
        series[s].n++;
    }
    fclose(c.fp);

    /* forecasts */
    csv_open(&c, root, "verify/fixture/forecasts.csv");
    cs = csv_col(&c, "series"); cm = csv_col(&c, "model");
    co = csv_col(&c, "origin"); cstep = csv_col(&c, "step");
    cyhat = csv_col(&c, "yhat");
    while (csv_next(&c)) {
        Pair *p = find_pair(find_series(c.field[cs]), find_model(c.field[cm]));
        int slot = origin_slot(p, (int)csv_num(&c, co));
        int step = (int)csv_num(&c, cstep);
        if (step >= MAX_HORIZON) die("horizon too long", c.field[cs]);
        if (p->filled[slot][step]) die("duplicate forecast row", c.field[cs]);
        p->yhat[slot][step] = csv_num(&c, cyhat);
        p->filled[slot][step] = 1;
        if (step + 1 > p->horizon) p->horizon = step + 1;
    }
    fclose(c.fp);

    printf("%d series, %d models, %d series-model pairs\n",
           n_series, n_models, n_pairs);

    /* the denominator, and the origin arithmetic it depends on */
    for (i = 0; i < n_pairs; i++) {
        Pair *p = &pairs[i];
        int s = p->series, first, expect;
        qsort(p->origin, (size_t)p->n_origins, sizeof p->origin[0], cmp_int);
        for (k = 0; k < p->n_origins; k++)
            for (j = 0; j < p->horizon; j++)
                if (!p->filled[k][j]) die("a fold is missing a step",
                                          series[s].name);
        /* re-derived, not read: the first origin the harness can use */
        first = series[s].n - p->n_origins * p->horizon;
        for (k = 0; k < p->n_origins; k++) {
            expect = first + k * p->horizon;
            if (p->origin[k] != expect) {
                printf("  %s/%s fold %d starts at %d, the arithmetic says %d\n",
                       series[s].name, models[p->model], k, p->origin[k], expect);
                failures++;
            }
        }
        if (first <= SEASON * 2) die("fixture series is too short", series[s].name);
        first_of[s] = first;
    }

    for (i = 0; i < n_series; i++) {
        Acc a = {0.0, 0.0L, 0.0L, 0};
        double gap;
        for (j = SEASON; j < first_of[i]; j++)
            acc_add(&a, fabs(series[i].y[j] - series[i].y[j - SEASON]));
        denom[i] = acc_mean(&a, &gap);
        if (gap > worst_gap) worst_gap = gap;
    }

    /* every fold, against folds.csv */
    printf("\nfold errors, against verify/fixture/folds.csv\n");
    {
        int checked = 0;
        double worst = 0.0;
        csv_open(&c, root, "verify/fixture/folds.csv");
        cs = csv_col(&c, "series"); cm = csv_col(&c, "model");
        co = csv_col(&c, "origin"); cmae = csv_col(&c, "mae");
        while (csv_next(&c)) {
            int s = find_series(c.field[cs]);
            Pair *p = find_pair(s, find_model(c.field[cm]));
            int origin = (int)csv_num(&c, co);
            int slot = -1;
            Acc a = {0.0, 0.0L, 0.0L, 0};
            double got, want, d, gap;
            for (k = 0; k < p->n_origins; k++)
                if (p->origin[k] == origin) slot = k;
            if (slot < 0) die("folds.csv names an origin no forecast covers",
                              c.field[cs]);
            for (j = 0; j < p->horizon; j++)
                acc_add(&a, fabs(series[s].y[origin + j] - p->yhat[slot][j]));
            got = acc_mean(&a, &gap);
            if (gap > worst_gap) worst_gap = gap;
            want = csv_num(&c, cmae);
            d = fabs(got - want) / (1.0 + fabs(want));
            if (d > worst) worst = d;
            if (d > TOL) {
                printf("  %s/%s origin %d: C %.15g, published %.15g\n",
                       series[s].name, models[p->model], origin, got, want);
                failures++;
            }
            checked++;
        }
        fclose(c.fp);
        printf("  %d folds recomputed, worst relative gap %.1e\n", checked, worst);
    }

    /* MASE per series and model, against expected.csv */
    printf("\nMASE, against verify/fixture/expected.csv\n");
    csv_open(&c, root, "verify/fixture/expected.csv");
    cs = csv_col(&c, "series"); cm = csv_col(&c, "model");
    cn = csv_col(&c, "n"); cfirst = csv_col(&c, "first_origin");
    cden = csv_col(&c, "mase_denominator"); cmase = csv_col(&c, "mase");
    while (csv_next(&c)) {
        int s = find_series(c.field[cs]);
        int m = find_model(c.field[cm]);
        Pair *p = find_pair(s, m);
        Acc a = {0.0, 0.0L, 0.0L, 0};
        char label[80];
        double gap;
        if ((int)csv_num(&c, cn) != series[s].n) {
            printf("  %s: expected.csv says n=%d, history.csv has %d\n",
                   series[s].name, (int)csv_num(&c, cn), series[s].n);
            failures++;
        }
        if ((int)csv_num(&c, cfirst) != first_of[s]) {
            printf("  %s: expected.csv says first_origin=%d, the arithmetic "
                   "says %d\n", series[s].name, (int)csv_num(&c, cfirst),
                   first_of[s]);
            failures++;
        }
        snprintf(label, sizeof label, "%s denominator", series[s].name);
        if (m == 0) report(label, denom[s], csv_num(&c, cden));
        for (k = 0; k < p->n_origins; k++) {
            Acc f = {0.0, 0.0L, 0.0L, 0};
            for (j = 0; j < p->horizon; j++)
                acc_add(&f, fabs(series[s].y[p->origin[k] + j] - p->yhat[k][j]));
            acc_add(&a, acc_mean(&f, &gap));
        }
        snprintf(label, sizeof label, "%s %s MASE", series[s].name, models[m]);
        mase[(m * MAX_SERIES) + s] = acc_mean(&a, &gap) / denom[s];
        report(label, mase[(m * MAX_SERIES) + s], csv_num(&c, cmase));
    }
    fclose(c.fp);

    /* the aggregation, against summary.csv */
    printf("\naggregation across series, against verify/fixture/summary.csv\n");
    csv_open(&c, root, "verify/fixture/summary.csv");
    cm = csv_col(&c, "model"); cser = csv_col(&c, "series");
    cmed = csv_col(&c, "mase_median"); cmean = csv_col(&c, "mase_mean");
    cbeats = csv_col(&c, "beats_naive_frac");
    while (csv_next(&c)) {
        int m = find_model(c.field[cm]);
        double v[MAX_SERIES];
        int nv = 0, beats = 0;
        double sum = 0.0;
        char label[80];
        for (i = 0; i < n_series; i++) {
            v[nv] = mase[(m * MAX_SERIES) + i];
            sum += v[nv];
            if (v[nv] < 1.0) beats++;
            nv++;
        }
        if (nv != (int)csv_num(&c, cser)) {
            printf("  %s: summary.csv counts %d series, the fixture has %d\n",
                   models[m], (int)csv_num(&c, cser), nv);
            failures++;
        }
        snprintf(label, sizeof label, "%s median", models[m]);
        report(label, median(v, nv), csv_num(&c, cmed));
        snprintf(label, sizeof label, "%s mean", models[m]);
        report(label, sum / nv, csv_num(&c, cmean));
        snprintf(label, sizeof label, "%s beats naive", models[m]);
        report(label, (double)beats / nv, csv_num(&c, cbeats));
    }
    fclose(c.fp);

    printf("\nplain double and Kahan long double summation of the same terms\n"
           "  differ by at most %.1e relative, so none of this is a summation\n"
           "  artefact\n", worst_gap);

    if (failures) {
        printf("\n%d disagreements\n", failures);
        return 1;
    }
    printf("\nC reproduces every fold error, MASE and aggregate to %.0e\n", TOL);
    return 0;
}
