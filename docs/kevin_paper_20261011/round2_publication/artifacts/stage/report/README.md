# Round2 repeated-benchmark comparison

All three runs are shown. The representative was selected only by the fixed validation criterion. Every test cohort was previously seen during project development; none is a fresh clinical test.

| Cohort | Old | Round1 c3 | Seed11 | Seed12 | Seed13 |
|---|---:|---:|---:|---:|---:|
| Full reused image test (5,450) | 99.93% | 99.87% | 99.96% | 99.93% | 99.96% |
| Reused same-case comparison (1,058) | 99.91% | 99.91% | 100.00% | 99.91% | 100.00% |
| Reused hash-unexposed subset (756) | 99.87% | 99.87% | 100.00% | 99.87% | 100.00% |
| Reused reader benchmark (100) | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

Reader accuracy: mean 39.67%; range 37.00–44.00% across these three fixed readers.

Intervals in analysis.json are conditional paired percentile intervals; hash groups are not patient identities, and ceiling/degenerate intervals do not remove dataset limitations. No outcome-based winner or clinical-readiness claim is made.

All figures require visual inspection before publication.
