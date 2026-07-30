# Extraction style ablation

All cells use the same frozen Strong Actor and matched 200k training budget. Each cell is `paired style shift / paired task retention`.

| Variant | Aggressive | Defensive | Explorer |
|---|---:|---:|---:|
| Full | +0.023 / 91.1% | +0.006 / 96.3% | +0.004 / 91.6% |
| Reward + KL | +0.017 / 92.1% | -0.031 / 92.1% | -0.031 / 87.9% |
| Reward only | +0.082 / 94.4% | +0.081 / 88.8% | -0.016 / 87.4% |

Style-shift metrics are style-specific and are not a cross-style ranking. Full gate values and disclosed failures are stored in [`style-ablation.json`](style-ablation.json).

`test_cases_accessed=false`; official-test cases were not opened.
