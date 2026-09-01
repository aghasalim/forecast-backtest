-- Recompute every fold error, every MASE and the whole aggregation in SQL.
--
-- The published figures come out of numpy inside src/fb/harness.py: one
-- expression for the denominator, one for the fold errors, one mean of means.
-- SQL has none of numpy's vectorised shortcuts, so getting the same answer here
-- means the joins line each forecast up with the observation it is supposed to
-- be predicting, independently of how the Python sliced its arrays.
--
-- Nothing is read from expected.csv except to compare against. first_origin,
-- the horizon and the number of origins are all derived from the data.
--
-- Run: sqlite3 -init verify/mase.sql :memory: ""
-- Every row it prints ends in ok or FAIL; verify/verify.sh checks for FAIL.

.mode csv
.headers off
.import --csv verify/fixture/history.csv   history
.import --csv verify/fixture/forecasts.csv forecasts
.import --csv verify/fixture/folds.csv     published_folds
.import --csv verify/fixture/expected.csv  published_expected
.import --csv verify/fixture/summary.csv   published_summary

CREATE TEMP VIEW h AS
    SELECT series, CAST(t AS INTEGER) AS t, CAST(y AS REAL) AS y FROM history;

CREATE TEMP VIEW f AS
    SELECT series, model, CAST(origin AS INTEGER) AS origin,
           CAST(step AS INTEGER) AS step, CAST(yhat AS REAL) AS yhat
    FROM forecasts;

-- the shape of the backtest, derived rather than read
CREATE TEMP VIEW shape AS
    SELECT f.series, f.model,
           (SELECT COUNT(*) FROM h WHERE h.series = f.series)      AS n,
           COUNT(DISTINCT f.origin)                                AS n_origins,
           COUNT(DISTINCT f.step)                                  AS horizon,
           (SELECT COUNT(*) FROM h WHERE h.series = f.series)
               - COUNT(DISTINCT f.origin) * COUNT(DISTINCT f.step) AS first_origin
    FROM f GROUP BY f.series, f.model;

-- mean |y[t] - y[t-168]| over t strictly before the first origin, so the
-- yardstick never overlaps the window being scored
CREATE TEMP VIEW denom AS
    SELECT s.series, s.model, AVG(ABS(a.y - b.y)) AS denom
    FROM shape s
    JOIN h a ON a.series = s.series AND a.t >= 168 AND a.t < s.first_origin
    JOIN h b ON b.series = s.series AND b.t = a.t - 168
    GROUP BY s.series, s.model;

CREATE TEMP VIEW fold_mae AS
    SELECT f.series, f.model, f.origin, AVG(ABS(h.y - f.yhat)) AS mae
    FROM f JOIN h ON h.series = f.series AND h.t = f.origin + f.step
    GROUP BY f.series, f.model, f.origin;

CREATE TEMP VIEW mase AS
    SELECT m.series, m.model, AVG(m.mae) / d.denom AS mase
    FROM fold_mae m JOIN denom d
      ON d.series = m.series AND d.model = m.model
    GROUP BY m.series, m.model, d.denom;

-- numpy's median: the middle value, or the mean of the two middle ones
CREATE TEMP VIEW agg AS
    WITH ranked AS (
        SELECT model, mase,
               ROW_NUMBER() OVER (PARTITION BY model ORDER BY mase) AS rn,
               COUNT(*)     OVER (PARTITION BY model)               AS cnt
        FROM mase)
    SELECT model,
           MAX(cnt)                                                 AS n,
           AVG(CASE WHEN rn IN ((cnt + 1) / 2, (cnt + 2) / 2)
                    THEN mase END)                                  AS mase_median,
           AVG(mase)                                                AS mase_mean,
           AVG(CASE WHEN mase < 1.0 THEN 1.0 ELSE 0.0 END)          AS beats
    FROM ranked GROUP BY model;

.headers on
.print
.print -- fold errors, MASE and the aggregation, SQL against the published fixture

SELECT 'fold ' || p.series || '/' || p.model || '@' || p.origin AS what,
       m.mae                                                    AS sql_value,
       CAST(p.mae AS REAL)                                       AS published,
       CASE WHEN ABS(m.mae - CAST(p.mae AS REAL))
                 <= 1e-12 * (1 + ABS(CAST(p.mae AS REAL)))
            THEN 'ok' ELSE 'FAIL' END                            AS status
FROM published_folds p
JOIN fold_mae m ON m.series = p.series AND m.model = p.model
               AND m.origin = CAST(p.origin AS INTEGER)
WHERE status = 'FAIL'
UNION ALL
SELECT 'folds compared: ' || COUNT(*), NULL, NULL,
       CASE WHEN COUNT(*) = (SELECT COUNT(*) FROM published_folds)
            THEN 'ok' ELSE 'FAIL' END
FROM published_folds p JOIN fold_mae m
  ON m.series = p.series AND m.model = p.model
 AND m.origin = CAST(p.origin AS INTEGER)
UNION ALL
SELECT 'first_origin ' || s.series || '/' || s.model,
       s.first_origin, CAST(e.first_origin AS INTEGER),
       CASE WHEN s.first_origin = CAST(e.first_origin AS INTEGER)
            THEN 'ok' ELSE 'FAIL' END
FROM shape s JOIN published_expected e
  ON e.series = s.series AND e.model = s.model
UNION ALL
SELECT 'denominator ' || d.series, d.denom, CAST(e.mase_denominator AS REAL),
       CASE WHEN ABS(d.denom - CAST(e.mase_denominator AS REAL))
                 <= 1e-12 * (1 + ABS(CAST(e.mase_denominator AS REAL)))
            THEN 'ok' ELSE 'FAIL' END
FROM denom d JOIN published_expected e
  ON e.series = d.series AND e.model = d.model
WHERE d.model = 'seasonal_naive'
UNION ALL
SELECT 'mase ' || m.series || '/' || m.model, m.mase, CAST(e.mase AS REAL),
       CASE WHEN ABS(m.mase - CAST(e.mase AS REAL))
                 <= 1e-12 * (1 + ABS(CAST(e.mase AS REAL)))
            THEN 'ok' ELSE 'FAIL' END
FROM mase m JOIN published_expected e
  ON e.series = m.series AND e.model = m.model
UNION ALL
SELECT 'median ' || a.model, a.mase_median, CAST(p.mase_median AS REAL),
       CASE WHEN ABS(a.mase_median - CAST(p.mase_median AS REAL))
                 <= 1e-12 THEN 'ok' ELSE 'FAIL' END
FROM agg a JOIN published_summary p ON p.model = a.model
UNION ALL
SELECT 'mean ' || a.model, a.mase_mean, CAST(p.mase_mean AS REAL),
       CASE WHEN ABS(a.mase_mean - CAST(p.mase_mean AS REAL))
                 <= 1e-12 THEN 'ok' ELSE 'FAIL' END
FROM agg a JOIN published_summary p ON p.model = a.model
UNION ALL
SELECT 'beats naive ' || a.model, a.beats, CAST(p.beats_naive_frac AS REAL),
       CASE WHEN ABS(a.beats - CAST(p.beats_naive_frac AS REAL))
                 <= 1e-12 THEN 'ok' ELSE 'FAIL' END
FROM agg a JOIN published_summary p ON p.model = a.model
UNION ALL
SELECT 'series counted ' || a.model, a.n, CAST(p.series AS INTEGER),
       CASE WHEN a.n = CAST(p.series AS INTEGER) THEN 'ok' ELSE 'FAIL' END
FROM agg a JOIN published_summary p ON p.model = a.model;
