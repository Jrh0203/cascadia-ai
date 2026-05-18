# Overnight Training Summary

## Experiment: Mid-Features FSP Reservoir

Architecture: 10,862 features → 512 → 64 → 1 (~21 MB, 5.6M params)
— all v3 features MINUS per-cell adjacency (34K features / 17.6M params removed).

Opponents: FSP reservoir = {random, scarcity, preference, mce93} + past iters.
Recipe: 5 iters × 100K games × 15 epochs, LR 1e-4 → 3e-5, ε=0.1.

## Per-Iter Training RMSE

| iter1 | RMSE 3.66 | 21M |
| iter2 | RMSE 3.67 | 21M |
| iter3 | RMSE 3.71 | 21M |
| iter4 | RMSE 3.76 | 21M |
| iter5 | RMSE 3.76 | 21M |

## HH Validation: iter3 vs 3×mce93 (N=52)
```
TOURNAMENT RESULTS
==================================================================================

Strategy       Games  WinRate  MeanRank MeanScore   Bonus     SE
----------------------------------------------------------------------------------
mce_new           52    15.4%      2.81     93.62   98.35   0.42
mce_anchor       156    28.2%      2.40     94.20   99.01   0.26
mce_anchor       156    28.2%      2.40     94.20   99.01   0.26
mce_anchor       156    28.2%      2.40     94.20   99.01   0.26

Strategy          Bear     Elk  Salmon    Hawk     Fox
------------------------------------------------------------
mce_new           9.81   13.12   12.13   11.92   13.37
mce_anchor       11.12   11.00   12.13   11.34   14.59
mce_anchor       11.12   11.00   12.13   11.34   14.59
mce_anchor       11.12   11.00   12.13   11.34   14.59

Strategy         Rank1   Rank2   Rank3   Rank4
--------------------------------------------------
mce_new          15.4%   21.2%   30.8%   32.7%
mce_anchor       28.2%   26.3%   23.1%   22.4%
```

## HH Validation: iter5 vs 3×mce93 (N=52)
```
TOURNAMENT RESULTS
==================================================================================

Strategy       Games  WinRate  MeanRank MeanScore   Bonus     SE
----------------------------------------------------------------------------------
mce_new           52    21.2%      2.52     94.46  100.00   0.35
mce_anchor       156    26.3%      2.49     94.36   98.90   0.24
mce_anchor       156    26.3%      2.49     94.36   98.90   0.24
mce_anchor       156    26.3%      2.49     94.36   98.90   0.24

Strategy          Bear     Elk  Salmon    Hawk     Fox
------------------------------------------------------------
mce_new           9.69   11.27   13.17   11.48   14.85
mce_anchor       11.37   11.12   11.79   11.51   14.43
mce_anchor       11.37   11.12   11.79   11.51   14.43
mce_anchor       11.37   11.12   11.79   11.51   14.43

Strategy         Rank1   Rank2   Rank3   Rank4
--------------------------------------------------
mce_new          21.2%   32.7%   19.2%   26.9%
mce_anchor       26.3%   22.4%   26.9%   24.4%
```

## 4-Way Diagnostic: mce93 vs mid_fsp_iter5 vs legacy_fsp_iter1 vs sym_pool_iter1
```
TOURNAMENT RESULTS
==================================================================================

Strategy       Games  WinRate  MeanRank MeanScore   Bonus     SE
----------------------------------------------------------------------------------
mce_mce93         52    50.0%      1.96     95.27   99.87   0.48
mce_mid5          52    25.0%      2.31     93.98   98.81   0.45
mce_legacy1       52     5.8%      2.98     92.31   97.04   0.44
mce_pool1         52    19.2%      2.75     92.67   97.60   0.50

Strategy          Bear     Elk  Salmon    Hawk     Fox
------------------------------------------------------------
mce_mce93        14.73    9.96   11.54    8.98   15.48
mce_mid5         12.60   11.08   13.10   10.44   13.25
mce_legacy1       7.54   12.71   12.04   13.79   13.65
mce_pool1         8.29   12.02   12.35   13.04   13.87

Strategy         Rank1   Rank2   Rank3   Rank4
--------------------------------------------------
mce_mce93        50.0%   19.2%   15.4%   15.4%
mce_mid5         25.0%   34.6%   25.0%   15.4%
```

Completed: Thu Apr 16 03:50:52 EDT 2026
