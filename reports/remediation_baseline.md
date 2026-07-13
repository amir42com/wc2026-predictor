# Pre-Remediation Baseline Manifest

Recorded: 2026-07-13
Branch: `fix/scientific-audit`
Branch point: `master` @ `2b1e5d8bd6c870b1e11501b10793966689d1ff21` (current HEAD at time of recording)

## Artifact hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `data/raw/results.csv` | `59f49de6055179ebe8c7f4d9f31b579c938f9eafd25e925abdffd91885f08e43` |
| `data/processed/features.csv` | `d2fefbc3195b19d91c87152d541434475a61a1c4b5a4d25ffba29b0d6381a8c6` |
| `data/processed/team_state.csv` | `05e9af1c55372386fe4c09ea1fa53ddbeddcf4a459ff1b62d8fa4d0489bc46da` |
| `data/processed/elo_ratings.csv` | `ce1bfc4b6ba22593399b7cf6d02aff82bdcd580bfe94968bf0c1360110aaf30f` |
| `models/xgb_wc2026.joblib` | `a81bca98bb7d8cd0225453e1b975365b40b59ee6de352728a04eaa2eec69178e` |
| `reports/backtest_by_tournament.csv` | `cd8f3e6fc78736e6b5f6d109d5fdd30dbe6e450becd64a904ee9c739aaf4a0ef` |
| `reports/backtest_metrics_summary.csv` | `157c9c6a6f6719efa3f38ea0ecd8f25e176f8888156a110725849e6de80adb09` |
| `reports/backtest_predictions.csv` | `b9b3f357cdb6d6934d620b963948f81f72b5adf6d72387977a70cbdda63d7cc8` |

## Published headline numbers (verbatim copy of reports/backtest_metrics_summary.csv)

```csv
tournament,model,n,accuracy,log_loss,brier,statistic,value,detail
WC 2014,Raw XGBoost,64,0.609375,0.918165,0.541073,,,
WC 2014,Blend,64,0.609375,0.920663,0.542548,,,
WC 2014,Elo,64,0.609375,0.938830,0.555480,,,
WC 2018,Raw XGBoost,64,0.546875,0.962694,0.569676,,,
WC 2018,Blend,64,0.546875,0.954941,0.565921,,,
WC 2018,Elo,64,0.578125,0.950053,0.564498,,,
WC 2022,Raw XGBoost,64,0.531250,1.047861,0.616569,,,
WC 2022,Blend,64,0.531250,1.040508,0.612005,,,
WC 2022,Elo,64,0.546875,1.041799,0.611098,,,
Combined,Raw XGBoost,192,0.562500,0.976240,0.575773,,,
Combined,Blend,192,0.562500,0.972037,0.573491,,,
Combined,Elo,192,0.578125,0.976894,0.577025,,,
Paired test (192),McNemar exact (Blend vs Elo),192,,,,p_value,0.581055,"b=5 (blend-only correct), c=8 (elo-only correct), discordant=13"
Paired test (192),Bootstrap logloss diff (Blend-Elo),192,,,,mean_diff,-0.004857,"95% CI [-0.0259, 0.0175], 10000 resamples, seed 12345"
```

---

Pre-remediation baseline. All values below this line are superseded by the remediation phases.
