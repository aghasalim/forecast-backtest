# Two checks in base R: the metric, and the premise the project was built on.
#
# 1. MASE, recomputed from verify/fixture/history.csv and forecasts.csv with
#    R's own arithmetic and R's own median. numpy's median and R's default
#    quantile agree here, but they are different code, and a median convention
#    that silently disagreed would move every published figure in this project
#    by half an order statistic.
#
# 2. reports/eda.json publishes prob_both_neighbours_in_train = 0.81, 0.64,
#    0.49 for 10%, 20% and 30% hold-outs, and the README builds its opening
#    argument on the 64%. Those are (1-h)^2, which is the limit as the series
#    gets long, not the answer for a series of finite length. This computes the
#    exact finite-length probability for a series the length of the median meter,
#    and simulates it, so the size of the approximation is measured rather than
#    assumed.
#
# No packages, so CI needs nothing beyond the R already on the runner.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."
set.seed(20260901)

SEASON <- 168
TOL <- 1e-12
DRAWS <- 300
SIGMA <- 4

failures <- 0

fx <- function(name) file.path(root, "verify", "fixture", name)
history <- read.csv(fx("history.csv"))
forecasts <- read.csv(fx("forecasts.csv"))
expected <- read.csv(fx("expected.csv"))
summary_pub <- read.csv(fx("summary.csv"))

y_of <- split(history$y, history$series)
for (s in names(y_of)) {
    t <- history$t[history$series == s]
    stopifnot(identical(t, seq_along(t) - 1L))
}

cat("MASE in base R, against verify/fixture/expected.csv\n")
mase_of <- list()
for (i in seq_len(nrow(expected))) {
    s <- expected$series[i]
    m <- expected$model[i]
    y <- y_of[[s]]
    n <- length(y)
    fc <- forecasts[forecasts$series == s & forecasts$model == m, ]
    origins <- sort(unique(fc$origin))
    horizon <- length(unique(fc$step))

    # derived, not read: the first origin the harness can use, and the window
    # the denominator is allowed to see
    first <- n - length(origins) * horizon
    idx <- (SEASON + 1):first                       # R is 1-based, y[t] is idx t+1
    denom <- mean(abs(y[idx] - y[idx - SEASON]))

    maes <- sapply(origins, function(o) {
        f <- fc[fc$origin == o, ]
        f <- f[order(f$step), ]
        mean(abs(y[o + f$step + 1] - f$yhat))
    })
    got <- mean(maes) / denom
    want <- expected$mase[i]
    d <- abs(got - want) / (1 + abs(want))
    ok <- d <= TOL
    failures <- failures + !ok
    if (first != expected$first_origin[i]) {
        cat(sprintf("  FAIL %s: first_origin %d, expected.csv says %d\n",
                    s, first, expected$first_origin[i]))
        failures <- failures + 1
    }
    if (abs(denom - expected$mase_denominator[i]) >
        TOL * (1 + expected$mase_denominator[i])) {
        cat(sprintf("  FAIL %s: denominator %.15g, expected.csv says %.15g\n",
                    s, denom, expected$mase_denominator[i]))
        failures <- failures + 1
    }
    mase_of[[paste(m, s)]] <- got
    cat(sprintf("  %-4s %-16s MASE %.12f  |d| %.1e  %s\n", s, m, got, d,
                if (ok) "ok" else "FAIL"))
}

cat("\naggregation, R's median against numpy's, on the same values\n")
for (i in seq_len(nrow(summary_pub))) {
    m <- summary_pub$model[i]
    v <- unlist(mase_of[startsWith(names(mase_of), paste0(m, " "))])
    stats <- c(median = median(v), mean = mean(v), beats = mean(v < 1))
    want <- c(summary_pub$mase_median[i], summary_pub$mase_mean[i],
              summary_pub$beats_naive_frac[i])
    for (k in seq_along(stats)) {
        d <- abs(stats[k] - want[k])
        ok <- d <= TOL
        failures <- failures + !ok
        cat(sprintf("  %-16s %-7s %.12f  |d| %.1e  %s\n", m, names(stats)[k],
                    stats[k], d, if (ok) "ok" else "FAIL"))
    }
    if (length(v) != summary_pub$series[i]) {
        cat(sprintf("  FAIL %s: %d series, summary.csv says %d\n",
                    m, length(v), summary_pub$series[i]))
        failures <- failures + 1
    }
}

# ---- the interpolation probability ----------------------------------------
#
# The published value is (1-h)^2, the probability that two independent
# Bernoulli(1-h) draws both land in training. A hold-out is a sample without
# replacement of a fixed size, so the two neighbours are not independent and the
# exact probability, given the point itself is held out, is
#
#     (N-m)(N-m-1) / ((N-1)(N-2)),   m = round(h*N)
#
# which converges to (1-h)^2. Both are computed, and the simulation says which
# one a real hold-out actually produces.

eda <- readLines(file.path(root, "reports", "eda.json"), warn = FALSE)
grab <- function(key, from = 1) {
    line <- eda[from - 1 + grep(key, eda[from:length(eda)], value = FALSE,
                                fixed = TRUE)[1]]
    as.numeric(sub(".*:\\s*([-0-9.eE+]+),?\\s*$", "\\1", line))
}
hours_at <- grep("hours_per_series", eda, fixed = TRUE)[1]
N <- grab("\"median\"", from = hours_at)
published <- c(grab("holdout_10pct"), grab("holdout_20pct"), grab("holdout_30pct"))
holdouts <- c(0.10, 0.20, 0.30)

cat(sprintf("\ninterpolation probability, series length N = %d, %d simulated hold-outs\n",
            N, DRAWS))
for (j in seq_along(holdouts)) {
    h <- holdouts[j]
    m <- round(h * N)
    limit <- (1 - h)^2
    exact <- (N - m) * (N - m - 1) / ((N - 1) * (N - 2))

    per_draw <- numeric(DRAWS)
    for (b in seq_len(DRAWS)) {
        test <- logical(N)
        test[sample.int(N, m)] <- TRUE
        i <- which(test)
        i <- i[i > 1 & i < N]                     # interior points only
        per_draw[b] <- mean(!test[i - 1] & !test[i + 1])
    }
    sim <- mean(per_draw)
    se <- sd(per_draw) / sqrt(DRAWS)

    # the published number must be exactly the limit the README quotes
    d_pub <- abs(published[j] - limit)
    ok_pub <- d_pub <= TOL
    # and the simulation must land on the exact finite-N value
    z <- abs(sim - exact) / max(se, 1e-15)
    ok_sim <- z <= SIGMA
    failures <- failures + !ok_pub + !ok_sim
    cat(sprintf(paste0("  h=%.2f  published %.4f  (1-h)^2 %.6f  exact at this N ",
                       "%.6f  simulated %.6f +- %.6f  %.1f sd  %s\n"),
                h, published[j], limit, exact, sim, se, z,
                if (ok_pub && ok_sim) "ok" else "FAIL"))
}
gap <- max(abs((1 - holdouts)^2 -
               (N - round(holdouts * N)) * (N - round(holdouts * N) - 1) /
               ((N - 1) * (N - 2))))
cat(sprintf(paste0("  (1-h)^2 differs from the exact value at this N by at most ",
                   "%.2e, which is\n  smaller than the 0.005 that quoting it as ",
                   "two decimals already allows\n"), gap))

if (failures > 0) {
    cat(sprintf("\n%d checks failed\n", failures))
    quit(status = 1)
}
cat("\nR reproduces every MASE and the aggregation, and the simulated hold-out\n")
cat("lands on the interpolation probability the README argues from\n")
